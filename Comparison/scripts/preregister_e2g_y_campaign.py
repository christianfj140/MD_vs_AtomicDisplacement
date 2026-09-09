#!/usr/bin/env python3
"""Freeze the direct E2g_y campaign protocol before running SIESTA.

The C3 symmetry shortcut (graphene_Gamma_sublattice_C3_symmetry_reduction)
was BLOCKED: the SCF k-point mesh is not C3-covariant (see
diagnose_c3_kmesh_and_field_covariance.py), so E2g_y is measured directly
instead of derived from the already-certified E2g_x. Same protocol as
delta_translation_campaign_preregistration_v1.json (h ladder, thresholds,
Richardson, fail-closed rules), applied to the collective E2g_y mode:
C1 = +Q/sqrt(2) y-hat, C2 = -Q/sqrt(2) y-hat.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/e2g_y_campaign_preregistration"
SIESTA_BINARY = Path("/home/christian/bin/siesta")
C3_DIAGNOSTIC = (
    REPO_ROOT / "Comparison/results/epc/c3_kmesh_and_field_covariance_diagnostic"
    / "c3_kmesh_and_field_covariance_diagnostic.json"
)
E2G_X_ARTIFACT = (
    REPO_ROOT / "Comparison/results/epc/graphene_gamma_sublattice_symmetry_reduction"
    / "graphene_gamma_sublattice_symmetry_reduction.json"
)

H_LADDER_ANG = [0.000625, 0.00125, 0.0025, 0.005, 0.01]
NEW_Q_ANG = [-0.02, -0.0025, -0.00125, -0.000625, 0.000625, 0.00125, 0.0025, 0.02, -0.01, -0.005, 0.005, 0.01]


def main() -> int:
    c3 = json.loads(C3_DIAGNOSTIC.read_text())
    e2g_x = json.loads(E2G_X_ARTIFACT.read_text())

    report = {
        "schema": "e2g_y_campaign_preregistration_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "reason_for_direct_campaign": (
            "graphene_Gamma_sublattice_C3_symmetry_reduction was BLOCKED: the real SCF "
            "k-point mesh is not C3-covariant (155/211 trimmed .KP points, 155/220 "
            "non-trimmed have no C3 partner), and the near-atomic-plane electrostatic "
            "potential shows a matching genuine C3 asymmetry. Not a bug in the symmetry "
            "operator construction (geometry, real-space mesh, S, orbital representation, "
            "D_E2g all verified C3-exact). Fail-closed decision: do not use the C3 shortcut."
        ),
        "c3_diagnostic_artifact_sha256": file_sha256(C3_DIAGNOSTIC),
        "e2g_x_artifact_sha256": file_sha256(E2G_X_ARTIFACT),
        "e2g_x_status": e2g_x["graphene_Gamma_E2g_x_delta_convergence"]["verdict"],
        "mode": {
            "definition": "collective, Frobenius-normalized: C1 = +Q/sqrt(2) y-hat, C2 = -Q/sqrt(2) y-hat",
            "name": "e2g_y",
        },
        "basis": {
            "name": "PROD-SZ", "central_dir": str(PROD_CENTRAL.relative_to(REPO_ROOT)),
            "c_ion_xml_sha256": file_sha256(PROD_CENTRAL / "C.ion.xml"),
            "c_psf_sha256": file_sha256(PROD_CENTRAL / "C.psf"),
        },
        "executable": {"path": str(SIESTA_BINARY), "sha256": file_sha256(SIESTA_BINARY)},
        "scf": {
            "dm_tolerance": "1.d-6", "mesh_cutoff_ry": 600.0,
            "kgrid_monkhorst_pack": [[20, 0, 0], [0, 20, 0], [0, 0, 1]],
        },
        "h_ladder_ang": H_LADDER_ANG,
        "decisive_window_ang": [0.005, 0.0025, 0.00125],
        "new_q_ang": NEW_Q_ANG,
        "n_new_siesta_runs": len(NEW_Q_ANG),
        "thresholds": {
            "e_rms_ev_per_ang": 5e-3, "e_rel": 0.01, "e_max_ev_per_ang": 15e-3,
            "combine_sources_in_quadrature": False,
        },
        "stencil": {"primary": "D5", "richardson": "R1=(16*E(0.0025)-E(0.005))/15, "
                    "R2=(16*E(0.00125)-E(0.0025))/15"},
        "amendment_policy": "No threshold, ladder, or run count in this file may change "
                             "after seeing SIESTA results from this campaign.",
        "not_repeated": "E2g_x (already PASS via the C2 symmetry reduction)",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "e2g_y_campaign_preregistration_v1.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"preregistered -> {path}")
    print(f"preregistration_sha256={file_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
