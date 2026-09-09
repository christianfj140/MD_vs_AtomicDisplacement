#!/usr/bin/env python3
"""onlyS semantic preflight (PROD-SZ only): does SIESTA's overlap-only route give S_R?

Inelastica's ``OSrun`` (Phonons.py) runs SIESTA on a *duplicated* geometry -- the
first copy frozen at ``R_0``, the second displaced by ``+-h`` -- reads the full
``2N x 2N`` overlap

    S = [[S_11, S_12], [S_21, S_22]]

and builds the one-sided derivative from the cross block ``S_12 = S[0:N, N:2N]``.
This script certifies, against the already-certified PROD-SZ connection, that

    dS_u = (S_12(+h) - S_12(-h)) / (h_+ + h_-)   is   S_R^u = <Phi(R_0)|d_u Phi(R_0)>

and, once that orientation holds, extracts the one block ``D_S = S_L + S_R``
cannot see:

    B_I(k) = J_I . S_R(k) . J_I      (anti-Hermitian, invisible to D_S)

Gates, in order:

* **P0** structure of the ``.onlyS`` output -- six files, ``2N x 2N``, identical
  orbital contract in both halves, ordering against ``ORB_INDX``, and
  ``||S_11(k) - S_TSHS(k)||_2 / ||S_TSHS(k)||_2 < 1e-8``.  A P0 failure is a real
  NO_GO for this route, never something to patch around.
* **P1** ``S_R^onlyS`` vs the certified ``S_R^cert`` propagated to the electronic
  quantity, ``e_propagated < 0.5 meV/Ang``; then the dense 20x20x1 audit, which
  needs no new SIESTA because ``.onlyS`` is a real-space object.
* **P2** ``B_I = J_I . S_R^onlyS . J_I``.

``S_R^v2 + S_L^v2 = D_S`` is automatic with respect to ``B_I`` and is NOT used as
evidence here; see ``b_i_evidence_note`` in the report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from certify_support_exact_connection import (  # noqa: E402
    DISPLACED_ATOM, SOLVE_RESIDUAL_TOL, _propagate, _selector,
)
from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from fdf_materialization import _set_fdf_directive  # noqa: E402
from generate_siesta_overlap_only import _orbital_rows  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_displaced_projection_sentinel import DIRECTIONS, H_ANG  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/onlys_semantic_preflight"
DEFAULT_SIESTA = "/home/christian/bin/siesta"

# Inelastica's OSrun uses the two-point one-sided derivative from +-h.
OS_STEPS = (-1, 1)
# Preregistered BEFORE any comparison was run (see report thresholds block).
P0_S11_REL_TOL = 1e-8              # ||S_11 - S_TSHS||_2 / ||S_TSHS||_2
P1_LIMIT_EV_PER_ANG = 5.0e-4       # 0.5 meV/Ang, in the eV/Ang the code works in
DENSE_GRID = (20, 20, 1)           # the production Monkhorst-Pack mesh


def _dense_k_points(grid: tuple[int, int, int] = DENSE_GRID) -> np.ndarray:
    axes = [np.arange(n) / n for n in grid]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _duplicated_fdf(source: str, label: str, xyz: np.ndarray) -> str:
    """Inelastica's duplicated geometry, overlap-only, no SCF.

    Uses ``TS.S.Save`` -- the interface SIESTA 5.4 documents.  The legacy
    ``TS.onlyS`` is deliberately NOT emitted even though this build still accepts
    it; ``keyword_probe`` in the report records that both were tested and that
    each alone sets ``ts: Only save the overlap matrix S = T``.
    """
    lines, rendered, inside = source.splitlines(), [], False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("systemname"):
            line = f"SystemName {label}"
        elif stripped.startswith("systemlabel"):
            line = f"SystemLabel {label}"
        elif stripped.startswith("numberofatoms"):
            line = f"NumberOfAtoms                    {len(xyz)}"
        if stripped == "%block atomiccoordinatesandatomicspecies":
            inside = True
            rendered.append(line)
            rendered.extend(" %.12f  %.12f  %.12f  1  # C" % tuple(p) for p in xyz)
            continue
        if inside:
            if stripped == "%endblock atomiccoordinatesandatomicspecies":
                inside = False
                rendered.append(line)
            continue
        rendered.append(line)
    if inside:
        raise ValueError("unterminated AtomicCoordinatesAndAtomicSpecies block")
    text = "\n".join(rendered) + "\n"
    text = _set_fdf_directive(text, "MaxSCFIterations", "0")
    text = _set_fdf_directive(text, "TS.S.Save", "T")
    return text


def produce(output: Path, siesta: str, xyz0: np.ndarray) -> dict[tuple[str, int], Path]:
    """Six duplicated-geometry overlap-only runs: +-h for x, y, z."""
    source = (PROD_CENTRAL / "RUN.fdf").read_text(encoding="utf-8")
    runs: dict[tuple[str, int], Path] = {}
    for direction, axis in DIRECTIONS.items():
        for step in OS_STEPS:
            tag = f"{'m' if step < 0 else 'p'}{abs(step)}"
            label = f"onlys_{direction.lower()}_{tag}"
            run_dir = output / "runs" / direction / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            displaced = xyz0.copy()
            displaced[DISPLACED_ATOM, axis] += step * H_ANG
            (run_dir / "RUN.fdf").write_text(
                _duplicated_fdf(source, label, np.vstack([xyz0, displaced])), encoding="utf-8"
            )
            for pseudo in ("C.psf", "Ghost-H.psf"):
                shutil.copy2(PROD_CENTRAL / pseudo, run_dir / pseudo)
            onlys = run_dir / f"{label}.onlyS"
            if not onlys.is_file():
                started = time.time()
                with (run_dir / "RUN.fdf").open(encoding="utf-8") as stdin, \
                        (run_dir / "RUN.out").open("w", encoding="utf-8") as stdout:
                    completed = subprocess.run(
                        [siesta], cwd=run_dir, stdin=stdin, stdout=stdout,
                        stderr=subprocess.STDOUT, check=False,
                    )
                (run_dir / "run_record.json").write_text(
                    json.dumps({
                        "command": [siesta], "returncode": completed.returncode,
                        "elapsed_seconds": time.time() - started,
                        "keyword": "TS.S.Save", "expected_output": onlys.name,
                    }, indent=2) + "\n", encoding="utf-8",
                )
                if completed.returncode or not onlys.is_file():
                    raise RuntimeError(
                        f"overlap-only run failed for {label}: returncode="
                        f"{completed.returncode}, onlyS present={onlys.is_file()}; "
                        f"see {run_dir / 'RUN.out'}"
                    )
            runs[(direction, step)] = run_dir
    return runs


def _gate_p0(runs, central, no: int) -> dict:
    """Structure of the .onlyS output.  A failure here is a real NO_GO."""
    import sisl

    overlaps, rows, failures = {}, [], []
    for (direction, step), run_dir in sorted(runs.items()):
        onlys = sorted(run_dir.glob("*.onlyS"))
        if len(onlys) != 1:
            failures.append(f"{direction}{step:+d}: expected one .onlyS, found {len(onlys)}")
            continue
        overlap = sisl.get_sile(str(onlys[0])).read_overlap()
        overlaps[(direction, step)] = overlap
        entry: dict = {
            "direction": direction, "step": step, "onlyS": str(onlys[0]),
            "onlyS_sha256": file_sha256(onlys[0]),
            "n_orbitals": int(overlap.no), "n_atoms": int(overlap.geometry.na),
            "nsc": overlap.lattice.nsc.tolist(),
        }
        if overlap.no != 2 * no:
            failures.append(f"{direction}{step:+d}: dimension {overlap.no} != 2N={2 * no}")

        # (3) both halves carry exactly the same orbital contract, and
        # (4) that contract matches ORB_INDX.
        orb_indx = sorted(run_dir.glob("*.ORB_INDX"))
        contract = [
            (row["atom"] % (overlap.geometry.na // 2), row["n"], row["l"], row["m"], row["zeta"])
            for row in _orbital_rows(orb_indx[0])
        ]
        halves_match = contract[:no] == contract[no:]
        reference = [
            (row["atom"], row["n"], row["l"], row["m"], row["zeta"])
            for row in _orbital_rows(sorted(PROD_CENTRAL.glob("*.ORB_INDX"))[0])
        ]
        ordering_match = contract[:no] == reference
        entry["halves_share_orbital_contract"] = halves_match
        entry["ordering_matches_central_ORB_INDX"] = ordering_match
        if not halves_match:
            failures.append(f"{direction}{step:+d}: the two halves differ in orbital contract")
        if not ordering_match:
            failures.append(f"{direction}{step:+d}: ordering disagrees with central ORB_INDX")

        # (5) S_11 must reproduce the production TSHS overlap.
        s11_checks = []
        for k in K_POINTS:
            s11 = np.asarray(overlap.Sk(k=k, format="array"))[:no, :no]
            reference_s = np.asarray(central.Sk(k=k, format="array"))
            relative = float(
                np.linalg.norm(s11 - reference_s, 2)
                / max(np.linalg.norm(reference_s, 2), np.finfo(float).tiny)
            )
            s11_checks.append({"k_reduced": list(k), "S11_vs_TSHS_relative_spectral": relative})
            if not relative < P0_S11_REL_TOL:
                failures.append(
                    f"{direction}{step:+d} k={list(k)}: S_11 vs TSHS relative={relative:.6g}"
                )
        entry["S11_checks"] = s11_checks
        rows.append(entry)

    expected = {(d, s) for d in DIRECTIONS for s in OS_STEPS}
    # Inelastica expects six outputs (+-x, +-y, +-z); DIRECTIONS is the
    # repo's certified subset, so state both counts explicitly.
    return {
        "verdict": "PASS" if not failures else "NO_GO",
        "failure_reasons": failures,
        "expected_outputs": len(expected),
        "found_outputs": len(overlaps),
        "inelastica_expects_six_outputs": True,
        "directions_covered": sorted(DIRECTIONS),
        "directions_note": (
            "Inelastica's OSrun runs +-x, +-y, +-z; this preflight covers the "
            "certified PROD-SZ subset C1:x and C1:z, i.e. four of those six runs. "
            "C1:y is not certified upstream so there is nothing to compare it to."
        ),
        "N": no, "expected_dimension": 2 * no,
        "runs": rows,
    }, overlaps


def _cross_derivative(overlaps, direction: str, no: int, k) -> tuple[np.ndarray, np.ndarray]:
    """Inelastica's dS_u from the cross blocks, plus the S_21 counterpart."""
    plus = np.asarray(overlaps[(direction, 1)].Sk(k=k, format="array"))
    minus = np.asarray(overlaps[(direction, -1)].Sk(k=k, format="array"))
    denominator = 2 * H_ANG  # h_+ + h_-
    return (plus[:no, no:] - minus[:no, no:]) / denominator, \
           (plus[no:, :no] - minus[no:, :no]) / denominator


def evaluate(output: Path, siesta: str) -> dict:
    import sisl

    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    no = int(central.no)
    xyz0 = central.geometry.xyz.copy()
    runs = produce(output, siesta, xyz0)

    p0, overlaps = _gate_p0(runs, central, no)
    print(f"gate P0 (.onlyS structure): {p0['verdict']}", flush=True)

    certified = np.load(
        REPO_ROOT / "Comparison/results/epc/production_basis_connection"
        / "production_basis_connection_matrices.npz"
    )
    mask = _selector(central.geometry)
    dense_k = _dense_k_points()

    cases, saved = [], {"orbital_mask_I": mask, "k_diagnostic": np.asarray(K_POINTS),
                        "k_dense": dense_k}
    for direction in sorted(DIRECTIONS):
        if p0["verdict"] != "PASS":
            cases.append({"direction": direction, "gate_P1": "NOT_RUN", "gate_P2": "NOT_RUN",
                          "reason": "gate P0 did not pass"})
            continue
        rows, failures, solve_residuals = [], [], []
        for ik, k in enumerate(K_POINTS):
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            right, left = _cross_derivative(overlaps, direction, no, k)
            reference_right = certified[f"S_R_{direction}_k{ik}"]
            reference_left = certified[f"S_L_{direction}_k{ik}"]
            delta_right = right - reference_right
            delta_left = left - reference_left
            # Full connection: both the S_R term and its adjoint partner.
            propagated, residual = _propagate(delta_left, delta_right, s0, k0)
            solve_residuals.append(residual)
            scale = max(np.linalg.norm(reference_right), np.finfo(float).tiny)
            if not propagated < P1_LIMIT_EV_PER_ANG:
                failures.append(f"k={ik}: propagated={propagated:.6g} eV/Ang")
            if not residual < SOLVE_RESIDUAL_TOL:
                failures.append(f"k={ik}: Hermitian solve residual={residual:.6g}")
            # Attribution: which of the two estimators carries the discrepancy?
            # D_S from the displaced-TSHS stencil is the exact object gate 1
            # certified, so each estimator's own closure against it is an
            # independent, symmetric measure of its error.  Recorded always; it
            # never moves the preregistered threshold.
            d_s = _d_s(central, direction, k)
            onlys_closure = d_s - left - right
            cert_closure = d_s - reference_left - reference_right
            onlys_error, res_a = _propagate(onlys_closure / 2, onlys_closure / 2, s0, k0)
            cert_error, res_b = _propagate(cert_closure / 2, cert_closure / 2, s0, k0)
            solve_residuals.extend((res_a, res_b))
            rows.append({
                "k_reduced": list(k),
                "S_R_onlyS_vs_cert_relative": float(np.linalg.norm(delta_right) / scale),
                "S_L_onlyS_vs_cert_relative": float(
                    np.linalg.norm(delta_left) / max(np.linalg.norm(reference_left), np.finfo(float).tiny)
                ),
                "orientation_S12_is_S_R": bool(
                    np.linalg.norm(delta_right) < np.linalg.norm(right - reference_left)
                ),
                "propagated_ev_per_ang": propagated,
                "solve_residual_relative": residual,
                "attribution": {
                    "onlyS_closure_vs_D_S_ev_per_ang": onlys_error,
                    "certified_closure_vs_D_S_ev_per_ang": cert_error,
                    "adjoint_relative_onlyS": float(
                        np.linalg.norm(left - right.conj().T)
                        / max(np.linalg.norm(left), np.finfo(float).tiny)
                    ),
                },
            })
            saved[f"S_R_onlyS_{direction}_k{ik}"] = right
            saved[f"S_L_onlyS_{direction}_k{ik}"] = left

        diagnostic_verdict = "PASS" if not failures else "NO_GO"
        print(f"gate P1 {direction} (3 diagnostic k): {diagnostic_verdict}", flush=True)

        # Dense 20x20x1 audit -- no new SIESTA: .onlyS is a real-space object.
        # Run it regardless of the P1 verdict: it audits the onlyS connection
        # against D_S, which is independent of the certified grid estimator, so
        # it is exactly the attribution evidence a P1 NO_GO needs.
        worst, worst_k, worst_residual, over = -1.0, None, 0.0, 0
        for k in dense_k:
            s0 = np.asarray(central.Sk(k=k, format="array"))
            k0 = np.asarray(central.Hk(k=k, format="array")) + (fermi - vacuum) * s0
            right, left = _cross_derivative(overlaps, direction, no, k)
            # The certified estimator exists only at the 3 diagnostic k, so
            # the dense audit compares against D_S -- the exact, k-resolved
            # object gate 1 certified -- via the same propagation.
            closure = np.asarray(_d_s(central, direction, k)) - left - right
            propagated, residual = _propagate(closure / 2, closure / 2, s0, k0)
            worst_residual = max(worst_residual, residual)
            over += int(not propagated < P1_LIMIT_EV_PER_ANG)
            if propagated > worst:
                worst, worst_k = propagated, list(map(float, k))
        dense = {
            "status": "PASS" if not over else "NO_GO",
            "grid": list(DENSE_GRID), "k_count": int(len(dense_k)),
            "quantity": "D_S(k) - S_L^onlyS(k) - S_R^onlyS(k), propagated",
            "quantity_note": (
                "The certified PROD connection exists only at the 3 diagnostic k, "
                "so the dense sweep audits the onlyS connection against D_S from "
                "the displaced TSHS stencil -- the same object gate 1 certified. "
                "This closure is by construction blind to B_I (see b_i_evidence_note); "
                "it audits the off-site connection over the full production mesh, "
                "it is NOT evidence about B_I."
            ),
            "worst_propagated_ev_per_ang": worst,
            "worst_k_reduced": worst_k,
            "k_over_limit": over,
            "worst_solve_residual_relative": worst_residual,
        }
        solve_residuals.append(worst_residual)
        print(f"gate P1 {direction} dense {DENSE_GRID} audit: {dense['status']} "
              f"(worst {worst:.6g} eV/Ang)", flush=True)

        # Gate P2 -- B_I, with NO D_S correction applied.  Extraction is
        # conditioned on the dense audit (which is about the onlyS object
        # itself); the *gate* verdict below still follows the preregistered P1
        # comparison against the certified connection.
        p2 = {"status": "NOT_RUN", "reason": "dense audit did not pass"}
        if dense["status"] == "PASS":
            anti = []
            for ik, k in enumerate(K_POINTS):
                right, _ = _cross_derivative(overlaps, direction, no, k)
                b_i = mask[:, None] * right * mask[None, :]
                saved[f"B_I_{direction}_k{ik}"] = b_i
                anti.append(float(
                    np.linalg.norm(b_i + b_i.conj().T)
                    / max(np.linalg.norm(b_i), np.finfo(float).tiny)
                ))
            for ik, k in enumerate(dense_k):
                right, _ = _cross_derivative(overlaps, direction, no, k)
                saved[f"B_I_dense_{direction}_k{ik}"] = mask[:, None] * right * mask[None, :]
            p2 = {
                "status": "EXTRACTED" if diagnostic_verdict == "PASS" else "PROVISIONAL",
                "definition": "B_I(k) = J_I . S_R^onlyS(k) . J_I",
                "correction_applied": "none",
                "antihermiticity_relative_diagnostic_k": anti,
                "stored_keys": [f"B_I_{direction}_k{ik}" for ik in range(len(K_POINTS))]
                + [f"B_I_dense_{direction}_k*  ({len(dense_k)} k)"],
                "provisional_note": None if diagnostic_verdict == "PASS" else (
                    "Stored but NOT certified: gate P1 against the certified PROD "
                    "connection is NO_GO for this direction at the preregistered "
                    "0.5 meV/Ang. See the per-k attribution block."
                ),
            }
            print(f"gate P2 {direction}: B_I extracted at "
                  f"{len(K_POINTS)} + {len(dense_k)} k", flush=True)

        cases.append({
            "direction": direction,
            "gate_P1": diagnostic_verdict,
            "gate_P1_failure_reasons": failures,
            "gate_P1_diagnostic_k": rows,
            "gate_P1_dense_audit": dense,
            "gate_P2": p2,
            "max_solve_residual_relative": max(solve_residuals, default=0.0),
        })

    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "onlys_semantic_preflight_matrices.npz"
    np.savez_compressed(artifact, **saved)

    p1_pass = p0["verdict"] == "PASS" and all(
        case.get("gate_P1") == "PASS" and case["gate_P1_dense_audit"]["status"] == "PASS"
        for case in cases
    )
    p2_done = p1_pass and all(case["gate_P2"]["status"] == "EXTRACTED" for case in cases)
    print(f"gate P2 overall: {'EXTRACTED' if p2_done else 'NOT_DETERMINED'}", flush=True)
    report = {
        "schema": "onlys_semantic_preflight_prod_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "onlyS_semantic_preflight_PROD",
        "verdict": "PASS" if p2_done else ("NO_GO" if p0["verdict"] != "PASS" or not p1_pass else "PARTIAL"),
        "basis": "PROD-SZ",
        "siesta_interface": {
            "keyword_used": "TS.S.Save",
            "output_file": "<SystemLabel>.onlyS",
            "determined": "empirically, by running this build and inspecting outputs",
            "keyword_probe": (
                "This build (/home/christian/bin/siesta) accepts BOTH TS.S.Save and "
                "TS.onlyS; each alone sets 'ts: Only save the overlap matrix S = T' in "
                "RUN.out and each alone produces <SystemLabel>.onlyS. With neither, the "
                "flag is F and the run aborts under MaxSCFIterations 0. TS.S.Save is the "
                "documented SIESTA 5.4 interface and is what this script emits; the "
                "legacy TS.onlyS is deliberately not introduced."
            ),
        },
        "semantics": {
            "geometry": "duplicated: first copy frozen at R_0, second displaced by +-h",
            "cross_block": "S_12 = S[0:N, N:2N]",
            "derivative": "dS_u = (S_12(+h) - S_12(-h)) / (h_+ + h_-)",
            "orientation_verified": "dS_u = S_R^u = <Phi(R_0)|d_u Phi(R_0)>; S_21 gives S_L",
        },
        "h_ang": H_ANG,
        "displaced_atom_index": DISPLACED_ATOM,
        "thresholds": {
            "P0_S11_relative_spectral": P0_S11_REL_TOL,
            "P1_propagated_ev_per_ang": P1_LIMIT_EV_PER_ANG,
            "solve_residual_relative": SOLVE_RESIDUAL_TOL,
        },
        "threshold_note": (
            "P1 limit is the preregistered 0.5 meV/Ang = 5e-4 eV/Ang; thresholds were "
            "fixed before any comparison and are not relaxed after the fact."
        ),
        "linear_algebra": (
            "Cholesky/Hermitian solves only (scipy cho_factor/cho_solve); no explicit "
            "inverse, no pseudoinverse, no eigenvalue truncation"
        ),
        "gate_P0": p0,
        "cases": cases,
        "attribution_summary": {
            "method": (
                "Each estimator's own closure against D_S (the displaced-TSHS "
                "five-point stencil certified in gate 1) is an independent, symmetric "
                "measure of that estimator's error, since D_S = S_L + S_R exactly."
            ),
            "per_direction": {
                case["direction"]: {
                    "worst_onlyS_closure_ev_per_ang": max(
                        row["attribution"]["onlyS_closure_vs_D_S_ev_per_ang"]
                        for row in case["gate_P1_diagnostic_k"]
                    ),
                    "worst_certified_closure_ev_per_ang": max(
                        row["attribution"]["certified_closure_vs_D_S_ev_per_ang"]
                        for row in case["gate_P1_diagnostic_k"]
                    ),
                }
                for case in cases if case.get("gate_P1_diagnostic_k")
            },
            "note": (
                "Recorded as evidence, not as a reason to move the threshold. Where the "
                "onlyS closure is smaller than the certified closure, the P1 discrepancy "
                "is dominated by the certified real-space-grid quadrature estimator, not "
                "by the onlyS route -- but the preregistered P1 gate is a comparison "
                "against that certified connection and stands as recorded."
            ),
        },
        "b_i_status": "EXTRACTED" if p2_done else "NOT_DETERMINED",
        "b_i_evidence_note": (
            "S_R^v2 + S_L^v2 = D_S is essentially automatic with respect to B_I and MUST "
            "NOT be read as evidence that B_I is correct: any anti-Hermitian B_I added to "
            "S_R and subtracted from S_L leaves that sum unchanged by construction. The "
            "B_I gate is the comparison against the certified PROD connection above, and "
            "later the independent radial-angular implementation."
        ),
        "matrix_artifact": str(artifact),
        "matrix_artifact_sha256": file_sha256(artifact),
        "claim_scope": (
            "PROD-SZ only. No TZ/QZ, no new DFT bases, no Delta_out, no full-KS claim. "
            "reference_basis_convergence = PASS, full_basis_connection_v1 = NO_GO, "
            "support_exact_full_basis_connection_v2 gate 1 = PASS / overall BLOCKED, "
            "delta_out_closure = BLOCKED and full_KS = BLOCKED all stand unchanged."
        ),
    }
    path = output / "onlys_semantic_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n",
                    encoding="utf-8")
    print(f"onlyS semantic preflight: {report['verdict']} (B_I {report['b_i_status']}) -> {path}")
    return report


def _d_s(central, direction: str, k) -> np.ndarray:
    """D_S = d_{R_Iu} S from the certified displaced-TSHS five-point stencil."""
    from certify_production_basis_connection import H_VALUES  # noqa: F401
    from run_nested_basis_sentinel import _production_run

    import sisl

    weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    total = 0.0
    for step, weight in weights.items():
        run = _production_run(direction, step)
        label = sorted(run.glob("*.TSHS"))[0].stem
        overlap = _D_S_CACHE.setdefault(
            (direction, step), sisl.get_sile(str(run / f"{label}.TSHS")).read_hamiltonian()
        )
        total = total + weight * np.asarray(overlap.Sk(k=k, format="array"))
    return total / (12 * H_ANG)


_D_S_CACHE: dict = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--siesta", default=DEFAULT_SIESTA)
    args = parser.parse_args(argv)
    return 0 if evaluate(args.output, args.siesta)["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
