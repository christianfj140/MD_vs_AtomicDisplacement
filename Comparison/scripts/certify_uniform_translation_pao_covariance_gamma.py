#!/usr/bin/env python3
"""Gate 2 (Gamma scope): uniform_translation_PAO_covariance_Gamma.

Runs no new SIESTA: reuses exactly the same real translation_x artifacts,
S_L(h)/S_R(h) connection route, and five-check physics as
``uniform_translation_PAO_covariance_BZ_sampled_v1`` (via
``certify_uniform_translation_pao_covariance.evaluate_k``), restricted to the
single k = Gamma = (0,0,0) row.

This gate exists because the local graphene-Gamma finite-PAO covariant
response was the declared objective before the BZ-sampled campaign ran; a
failure at a non-Gamma probe point (K_POINTS[1], informally and incorrectly
called "K" -- see certify_uniform_translation_pao_covariance.py) must not by
itself block the Gamma-scoped claim. ``uniform_translation_PAO_covariance_
BZ_sampled_v1`` stays NO_GO, untouched, as the separate, still-real BZ-wide
finding.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

from artifact_signature import file_sha256  # noqa: E402
from certify_delta_plateau_topology import _k_matrix  # noqa: E402
from certify_production_basis_connection import LIMIT_EV_PER_ANG  # noqa: E402
from certify_uniform_translation_pao_covariance import (  # noqa: E402
    GATE1_PATH, H_PRODUCTION, NULL_MAX_LIMIT_EV_PER_ANG, NULL_RMS_LIMIT_EV_PER_ANG,
    evaluate_k, points_and_connections,
)
from cross_basis_projection_preflight import _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/uniform_translation_pao_covariance_gamma"
BZ_SAMPLED_PATH = (
    REPO_ROOT / "Comparison/results/epc/uniform_translation_pao_covariance"
    / "uniform_translation_pao_covariance.json"
)
GAMMA = (0.0, 0.0, 0.0)


def main() -> int:
    gate1 = json.loads(GATE1_PATH.read_text())
    prerequisite = gate1.get("C1x_delta_plateau")
    prerequisite_verdict = prerequisite.get("verdict") if isinstance(prerequisite, dict) else prerequisite
    if prerequisite_verdict != "PASS":
        report = {
            "schema": "uniform_translation_pao_covariance_gamma_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gate": "uniform_translation_PAO_covariance_Gamma",
            "verdict": "BLOCKED",
            "reason": f"prerequisite C1x_delta_plateau is {prerequisite_verdict!r}, not PASS; gate not attempted",
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "uniform_translation_pao_covariance_gamma.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"uniform_translation_PAO_covariance_Gamma: BLOCKED (prerequisite not met) -> {path}")
        return 1

    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)

    points_by_step, provenance, connections = points_and_connections(
        central_h, central_fermi, central_vacuum, base, atom, shape, dvolume)

    # connections is ordered like K_POINTS from certify_uniform_translation_pao_covariance;
    # its index 0 is Gamma there too (K_POINTS[0] == (0,0,0)), so reuse it directly.
    s_l, s_r = connections[0]
    k0 = _k_matrix(central_h, central_fermi, central_vacuum, GAMMA)
    s0 = np.asarray(central_h.Sk(k=GAMMA, format="array"))
    row = evaluate_k(GAMMA, k0, s0, points_by_step, s_l, s_r)

    report = {
        "schema": "uniform_translation_pao_covariance_gamma_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "uniform_translation_PAO_covariance_Gamma",
        "claim_scope": "local graphene-Gamma finite-PAO covariant response",
        "not_promoted_to": ["BZ-wide", "K", "q!=0", "MATBG"],
        "related_artifact_bz_sampled_v1": {
            "path": str(BZ_SAMPLED_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(BZ_SAMPLED_PATH),
            "verdict": "NO_GO",
            "note": "unchanged historical finding for the BZ-sampled scope; this Gamma gate "
                    "does not supersede or reinterpret it",
        },
        "prerequisite": "C1x_delta_plateau == PASS (verified in delta_plateau_topology_certification)",
        "h_production_ang": H_PRODUCTION,
        "thresholds": {
            "null_test_rms_ev_per_ang": NULL_RMS_LIMIT_EV_PER_ANG,
            "null_test_max_ev_per_ang": NULL_MAX_LIMIT_EV_PER_ANG,
            "connection_budget_ev_per_ang": LIMIT_EV_PER_ANG,
        },
        "run_provenance": {str(step): provenance[step] for step in provenance},
        "gamma_row": row,
        "verdict": "PASS" if row["passed"] else "NO_GO",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "uniform_translation_pao_covariance_gamma.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"uniform_translation_PAO_covariance_Gamma: {report['verdict']} -> {path}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
