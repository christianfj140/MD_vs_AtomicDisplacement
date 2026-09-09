#!/usr/bin/env python3
"""Freeze the delta-plateau + translation_x campaign protocol before running SIESTA.

Writes ``delta_translation_campaign_preregistration_v1.json``: every hash,
threshold and rule the two pending gates (``delta_plateau_topology_certification``
and ``uniform_translation_PAO_covariance``) must be judged against. Nothing in
this file may change after seeing the SIESTA results -- that is the point of
pre-registering it.
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

OUTPUT = REPO_ROOT / "Comparison/results/epc/delta_translation_campaign_preregistration"
SIESTA_BINARY = Path("/home/christian/bin/siesta")
GAUGE_CONTRACT = REPO_ROOT / "Comparison/results/epc/pao_flow_audit/graph2mat_target_gauge_contract.json"
TOPOLOGY_PREFLIGHT = (
    REPO_ROOT / "Comparison/results/epc/delta_translation_topology_preflight"
    / "delta_translation_topology_preflight.json"
)

# h ladder = fd_perturbation_space.delta_sweep(center_ang=0.0025, ratio=2.0, count=5)
DELTA_LADDER_ANG = [0.000625, 0.00125, 0.0025, 0.005, 0.01]
REUSED_C1X_ANG = [-0.01, -0.005, 0.005, 0.01]
NEW_C1X_ANG = [-0.02, -0.0025, -0.00125, -0.000625, 0.000625, 0.00125, 0.0025, 0.02]
NEW_TRANSLATION_X_ANG = [-0.01, -0.005, 0.005, 0.01]


def main() -> int:
    gauge = json.loads(GAUGE_CONTRACT.read_text())
    topology = json.loads(TOPOLOGY_PREFLIGHT.read_text())
    if topology["verdict"] != "PASS":
        raise RuntimeError("topology preflight must be PASS before pre-registering this campaign")
    if gauge.get("verdict") != "PASS":
        raise RuntimeError("graph2mat_target_gauge_contract must be PASS before pre-registering this campaign")

    report = {
        "schema": "delta_translation_campaign_preregistration_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "basis": {
            "name": "PROD-SZ",
            "central_dir": str(PROD_CENTRAL.relative_to(REPO_ROOT)),
            "c_ion_xml_sha256": file_sha256(PROD_CENTRAL / "C.ion.xml"),
            "c_psf_sha256": file_sha256(PROD_CENTRAL / "C.psf"),
            "orb_indx_sha256": file_sha256(PROD_CENTRAL / "prod_central.ORB_INDX"),
        },
        "executable": {
            "path": str(SIESTA_BINARY),
            "sha256": file_sha256(SIESTA_BINARY),
        },
        "scf": {
            "dm_tolerance": "1.d-6",
            "mesh_cutoff_ry": 600.0,
            "kgrid_monkhorst_pack": [[20, 0, 0], [0, 20, 0], [0, 0, 1]],
            "xc_functional": "GGA",
            "xc_authors": "PBE",
        },
        "gauge": {
            "certified_by": str(GAUGE_CONTRACT.relative_to(REPO_ROOT)),
            "gauge_contract_sha256": file_sha256(GAUGE_CONTRACT),
            "verdict": gauge["verdict"],
        },
        "topology_preflight": {
            "artifact": str(TOPOLOGY_PREFLIGHT.relative_to(REPO_ROOT)),
            "artifact_sha256": file_sha256(TOPOLOGY_PREFLIGHT),
            "verdict": topology["verdict"],
        },
        "directions": {
            "C1_x": "single-atom (atom 0) displacement along x, delta-plateau ladder",
            "translation_x": "uniform whole-cell displacement along x, translation gate only",
        },
        "delta_ladder_ang": DELTA_LADDER_ANG,
        "decisive_window_ang": [0.005, 0.0025, 0.00125],
        "diagnostic_only_ang": {
            "0.01": "truncation-regime probe, not required to improve monotonically",
            "0.000625": "noise-floor probe, not required to improve monotonically",
        },
        "stencil": {
            "primary": "D5",
            "formula": "D5(h) f = (f(-2h) - 8 f(-h) + 8 f(h) - f(2h)) / (12 h)",
            "secondary_diagnostic": "D2",
            "secondary_formula": "D2(h) f = (f(h) - f(-h)) / (2 h)",
        },
        "richardson": {
            "formula": "Delta_R(h/2) = (16 * Delta(h/2) - Delta(h)) / 15",
            "applies_to": ["0.005 -> 0.0025", "0.0025 -> 0.00125"],
        },
        "runs": {
            "c1_x_reused_ang": REUSED_C1X_ANG,
            "c1_x_reused_dirs": [
                str((PROD_CENTRAL.parent / "C1_x" / tag).relative_to(REPO_ROOT))
                for tag in ("m2", "m1", "p1", "p2")
            ],
            "c1_x_new_ang": NEW_C1X_ANG,
            "translation_x_new_ang": NEW_TRANSLATION_X_ANG,
            "translation_x_full_ladder": False,
            "translation_y": False,
            "c1_y": False,
            "qneq0_or_matbg": False,
        },
        "thresholds": {
            "e_rms_ev_per_ang": 5e-3,
            "e_rel": 0.01,
            "e_max_ev_per_ang": 15e-3,
            "connection_budget_ev_per_ang": 1e-3,
            "combine_sources_in_quadrature": False,
        },
        "delta_plateau_topology_certification": {
            "object": "Delta_PAO_cov(h) = D5 K(h) - S_L(h) S^-1 K0 - K0 S^-1 S_R(h), recomputed per h "
                       "(S_L, S_R are NOT frozen at h=0.005)",
            "topology_rule": "TSHS support hash {(i,j,R)} must be identical across all h and signs; "
                              "any mismatch is NO_GO regardless of matrix convergence, and the "
                              "offending point is not dropped after the fact",
        },
        "uniform_translation_pao_covariance": {
            "runs_after_prerequisite": "only executed if delta_plateau_topology_certification == PASS",
            "checks": [
                "D5 K =~ 0",
                "D5 S =~ 0",
                "D5 S - (S_L + S_R) =~ 0",
                "Delta_PAO_cov - (A S^-1 K - K S^-1 A) =~ 0, with A = S_R =~ -S_L",
                "diag(C^dagger Delta_PAO_cov C) =~ 0 (electronic eigenbasis; interband elements NOT required to vanish)",
            ],
            "incorrect_test_explicitly_rejected": "||Delta_PAO_cov|| ~ 0 as a full matrix (wrong for an interband response)",
        },
        "unaffected_by_this_campaign": {
            "checkpoint_lineage": "NO_GO",
            "fine_tuning_candidate": "BLOCKED",
            "delta_out_closure": "NO_GO",
            "full_KS": "BLOCKED",
        },
        "amendment_policy": "No threshold, ladder, direction, or run count in this file may change "
                             "after seeing SIESTA results from this campaign.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "delta_translation_campaign_preregistration_v1.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"preregistered -> {path}")
    print(f"preregistration_sha256={file_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
