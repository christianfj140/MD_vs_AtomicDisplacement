#!/usr/bin/env python3
"""Gate 2: uniform_translation_PAO_covariance.

Runs only if ``delta_plateau_topology_certification``'s ``C1x_delta_plateau``
is PASS (Gate 1's own JSON is read as the prerequisite, not re-derived here).
Uses only the production stencil h=0.005 (points at +-0.005, +-0.010 Ang), on
the ``translation_x`` direction.

Does NOT require ``Delta_PAO_cov == 0`` as a full matrix -- that is the wrong
null test for an interband response (a rigid translation only forces the
*diagonal* of the electronic-basis representation, and the commutator
``A S^-1 K - K S^-1 A`` with ``A = S_R``, to vanish; off-diagonal/interband
elements are free). The checks are:

  D5 K_trans =~ 0                                  (electronic units)
  D5 S_trans =~ 0                                  (propagated to electronic
  D5 S_trans =~ S_L + S_R                           units via the certified
  S_L + S_R =~ 0  (rigid-translation identity)      connection budget, not a
                                                     raw 1/Ang comparison)
  Delta_PAO_cov_trans - (A S^-1 K0 - K0 S^-1 A) =~ 0   (electronic units)
  diag(C^dagger Delta_PAO_cov_trans C) =~ 0            (electronic units,
                                                         absolute budget, no
                                                         1% relative test on a
                                                         quantity whose exact
                                                         expected value is 0)

S_L(trans)/S_R(trans) reuse the exact same certified grid-quadrature algorithm
as ``certify_production_basis_connection._connections`` (same D5 stencil, same
``_orbital_grid``/``_bloch_values`` leaf functions), generalized only in which
atoms carry a displacement component -- the physics/algorithm is not
reimplemented, only its direction input.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256  # noqa: E402
from certify_delta_plateau_topology import (  # noqa: E402
    D5_WEIGHTS, PROD_SZ_ROOT, _d5, _k_matrix, _point, _rms,
)
from certify_full_basis_connection_semantic_v5 import solve  # noqa: E402
from certify_production_basis_connection import LIMIT_EV_PER_ANG, _propagate  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry, _bloch_values, _orbital_grid  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/uniform_translation_pao_covariance"
GATE1_PATH = (
    REPO_ROOT / "Comparison/results/epc/delta_plateau_topology_certification"
    / "delta_plateau_topology_certification.json"
)
H_PRODUCTION = 0.005
NULL_RMS_LIMIT_EV_PER_ANG = 5e-3
NULL_MAX_LIMIT_EV_PER_ANG = 15e-3


def _translation_geometry(base, atom, direction: fdp.Direction, displacement: float):
    xyz = base.xyz + displacement * direction.vectors
    return sisl.Geometry(xyz, atoms=[atom] * base.na, lattice=base.lattice)


def _translation_connections(direction: fdp.Direction, h: float, base, atom, shape, dvolume, k_points):
    """Same algorithm as certify_production_basis_connection._connections, generalized
    to an arbitrary multi-atom Direction instead of the hardcoded single-atom one-hot."""
    geometries = {step: _translation_geometry(base, atom, direction, step * h) for step in (-2, -1, 0, 1, 2)}
    grids = {step: _orbital_grid(geometry, shape) for step, geometry in geometries.items()}
    result = []
    for k in k_points:
        x0 = _bloch_values(*grids[0], geometries[0].no, k)
        right = 0.0
        left = 0.0
        for step, weight in D5_WEIGHTS.items():
            xj = _bloch_values(*grids[step], geometries[step].no, k)
            right = right + weight * np.asarray((x0.conj().T @ xj).toarray()) * dvolume
            left = left + weight * np.asarray((xj.conj().T @ x0).toarray()) * dvolume
        result.append((left / (12 * h), right / (12 * h)))
    return result


def _null_check(matrix: np.ndarray, overlap: np.ndarray) -> dict:
    rms, mx = _rms(matrix, overlap)
    return {"rms_ev_per_ang": rms, "max_ev_per_ang": mx,
            "passed": rms < NULL_RMS_LIMIT_EV_PER_ANG and mx < NULL_MAX_LIMIT_EV_PER_ANG}


def evaluate_k(k, k0: np.ndarray, s0: np.ndarray, points_by_step: dict, s_l: np.ndarray, s_r: np.ndarray) -> dict:
    """The five checks (D5K null, D5S-vs-connection, S_L+S_R null, commutator, electronic
    diagonal null) at one k-point. Shared by the BZ-sampled gate and the Gamma-only gate
    so the physics is computed once, not reimplemented per scope."""
    k_by_step = {step: _k_matrix(*points_by_step[step], k) for step in points_by_step}
    s_by_step = {step: np.asarray(points_by_step[step][0].Sk(k=k, format="array")) for step in points_by_step}
    d5k_trans = _d5(k_by_step, H_PRODUCTION)
    d5s_trans = _d5(s_by_step, H_PRODUCTION)

    d5k_null = _null_check(d5k_trans, s0)

    residual_d5s_vs_connection = d5s_trans - (s_l + s_r)
    ds_vs_connection_ev = _propagate(residual_d5s_vs_connection / 2, residual_d5s_vs_connection / 2, s0, k0)
    ds_vs_connection = {"propagated_ev_per_ang": ds_vs_connection_ev, "budget_ev_per_ang": LIMIT_EV_PER_ANG,
                         "passed": ds_vs_connection_ev < LIMIT_EV_PER_ANG}

    sl_plus_sr = s_l + s_r
    sl_plus_sr_ev = _propagate(sl_plus_sr / 2, sl_plus_sr / 2, s0, k0)
    sl_plus_sr_check = {"propagated_ev_per_ang": sl_plus_sr_ev, "budget_ev_per_ang": LIMIT_EV_PER_ANG,
                         "passed": sl_plus_sr_ev < LIMIT_EV_PER_ANG}

    left_term, _ = solve(s0, k0)
    right_term, _ = solve(s0, s_r)
    delta_pao_cov_trans = d5k_trans - s_l @ left_term - k0 @ right_term

    a = s_r
    a_left, _ = solve(s0, k0)
    a_right, _ = solve(s0, a)
    commutator = a @ a_left - k0 @ a_right
    commutator_residual = delta_pao_cov_trans - commutator
    commutator_null = _null_check(commutator_residual, s0)

    eigvals, c = eigh(k0, s0)
    diag_electronic = np.diag(c.conj().T @ delta_pao_cov_trans @ c)
    diag_matrix = np.diag(diag_electronic)
    diag_null = _null_check(diag_matrix, s0)

    row_pass = (d5k_null["passed"] and ds_vs_connection["passed"] and sl_plus_sr_check["passed"]
                and commutator_null["passed"] and diag_null["passed"])
    return {
        "k_reduced": list(k), "passed": row_pass,
        "D5K_trans_null": d5k_null,
        "D5S_trans_vs_S_L_plus_S_R": ds_vs_connection,
        "S_L_plus_S_R_null": sl_plus_sr_check,
        "commutator_residual_null": commutator_null,
        "diag_electronic_null": diag_null,
        "diag_max_abs_ev_per_ang": float(np.abs(diag_electronic).max()),
    }


def points_and_connections(central_h, central_fermi, central_vacuum, base, atom, shape, dvolume):
    """Real SIESTA points at the four translation_x geometries, plus S_L(h)/S_R(h) via
    the certified connection route -- shared setup for any scope (BZ-sampled or Gamma-only)."""
    direction = fdp.uniform_translation(base.na, "x")
    points_by_step, provenance = {}, {}
    for step in D5_WEIGHTS:
        run_dir = PROD_SZ_ROOT / "translation_x" / f"d0p{'01' if abs(step) == 2 else '005'}_{'plus' if step > 0 else 'minus'}"
        hamiltonian, fermi, vacuum, tshs_path = _point(run_dir)
        points_by_step[step] = hamiltonian, fermi, vacuum
        provenance[step] = {"run_dir": str(run_dir.relative_to(REPO_ROOT)), "tshs_sha256": file_sha256(tshs_path)}
    connections = _translation_connections(direction, H_PRODUCTION, base, atom, shape, dvolume, K_POINTS)
    return points_by_step, provenance, connections


def main() -> int:
    gate1 = json.loads(GATE1_PATH.read_text())
    prerequisite = gate1.get("C1x_delta_plateau")
    prerequisite_verdict = prerequisite.get("verdict") if isinstance(prerequisite, dict) else prerequisite
    if prerequisite_verdict != "PASS":
        report = {
            "schema": "uniform_translation_pao_covariance_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gate": "uniform_translation_PAO_covariance",
            "verdict": "BLOCKED",
            "reason": f"prerequisite C1x_delta_plateau is {prerequisite_verdict!r}, not PASS; gate not attempted",
            "gate1_artifact_sha256": file_sha256(GATE1_PATH),
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "uniform_translation_pao_covariance.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"uniform_translation_PAO_covariance: BLOCKED (prerequisite not met) -> {path}")
        return 1

    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)

    points_by_step, provenance, connections = points_and_connections(
        central_h, central_fermi, central_vacuum, base, atom, shape, dvolume)

    per_k = []
    overall_pass = True
    for ik, k in enumerate(K_POINTS):
        k0 = _k_matrix(central_h, central_fermi, central_vacuum, k)
        s0 = np.asarray(central_h.Sk(k=k, format="array"))
        s_l, s_r = connections[ik]
        row = evaluate_k(k, k0, s0, points_by_step, s_l, s_r)
        overall_pass &= row["passed"]
        per_k.append(row)

    report = {
        "schema": "uniform_translation_pao_covariance_bz_sampled_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "uniform_translation_PAO_covariance_BZ_sampled_v1",
        "gate_scope_note": (
            "Adjudicated over the K_POINTS conditioning sample (Gamma + two generic "
            "probes), not restricted to Gamma. Historically this artifact's `gate` field "
            "read plain 'uniform_translation_PAO_covariance'; renamed here to make explicit "
            "that a failure at a non-Gamma point does not, by itself, block a Gamma-only "
            "claim -- see certify_uniform_translation_pao_covariance_gamma.py. The verdict "
            "and every number below are unchanged by this rename."
        ),
        "k_points_label_correction": (
            "K_POINTS[1] = (1/3, 1/3, 0) was informally called 'K' while investigating this "
            "gate's NO_GO. Direct diagonalisation shows it has a minimum band gap of ~1.15 eV: "
            "it is NOT the Dirac point for this cell. True K/K' are at (1/3, 2/3, 0) / "
            "(2/3, 1/3, 0); see cross_basis_projection_preflight.K_POINTS_HISTORICAL_LABELS "
            "and Comparison/results/epc/translation_k_point_residual_diagnostic/."
        ),
        "prerequisite": "C1x_delta_plateau == PASS (verified above)",
        "h_production_ang": H_PRODUCTION,
        "thresholds": {
            "null_test_rms_ev_per_ang": NULL_RMS_LIMIT_EV_PER_ANG,
            "null_test_max_ev_per_ang": NULL_MAX_LIMIT_EV_PER_ANG,
            "connection_budget_ev_per_ang": LIMIT_EV_PER_ANG,
            "note": "null tests use an absolute budget only; no 1% relative criterion is "
                    "applied to a quantity whose exact expected value is zero",
        },
        "incorrect_test_explicitly_rejected": "||Delta_PAO_cov_trans|| ~ 0 as a full matrix",
        "run_provenance": {str(step): provenance[step] for step in provenance},
        "per_k": per_k,
        "verdict": "PASS" if overall_pass else "NO_GO",
        "unaffected_by_this_gate": {
            "checkpoint_lineage": "NO_GO", "fine_tuning_candidate": "BLOCKED",
            "delta_out_closure": "NO_GO", "full_KS": "BLOCKED",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "uniform_translation_pao_covariance.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"uniform_translation_PAO_covariance: {report['verdict']} -> {path}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
