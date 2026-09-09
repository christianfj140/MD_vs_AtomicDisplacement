#!/usr/bin/env python3
"""Prospective topology preflight for the delta-ladder and translation_x campaigns.

Checks, without running SIESTA, whether the neighbour topology of PROD-SZ's
2-atom central cell would change under any displacement the two pending gates
need: ``delta_plateau_topology_certification`` (C1:x, five h amplitudes) and
``uniform_translation_PAO_covariance`` (translation_x, h=0.005/0.010 only).
A topology change at some amplitude must be reported, not silently dropped --
see ``fd_perturbation_space.topology_margin``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

import fd_perturbation_space as fdp  # noqa: E402
import orbital_contract as oc  # noqa: E402
from artifact_signature import input_signature_sha256  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/delta_translation_topology_preflight"
BOHR_TO_ANG = 0.529177210903


def main() -> int:
    central = sisl.get_sile(str(PROD_CENTRAL / "RUN.fdf")).read_geometry()
    positions_ang = central.xyz.tolist()
    lattice_ang = central.cell.tolist()

    contract = oc.read_ion_xml(str(PROD_CENTRAL / "C.ion.xml"))
    rc_ang = max(shell["cutoff_bohr"] for shell in contract["shells"]) * BOHR_TO_ANG
    cutoff_ang = [rc_ang] * central.na

    directions = {
        "C1_x": fdp.one_hot(central.na, 0, "x"),
        "translation_x": fdp.uniform_translation(central.na, "x"),
    }

    # h = {0.000625, 0.00125, 0.0025, 0.005, 0.01} Ang -- the delta-plateau
    # ladder. A D5 stencil at each h needs +-h and +-2h, so the unique
    # displacement magnitudes to check are the ladder itself union its doubles.
    ladder = fdp.delta_sweep(center_ang=0.0025, ratio=2.0, count=5)
    stencil_magnitudes = sorted({round(h, 10) for h in ladder} | {round(2.0 * h, 10) for h in ladder})

    cases = []
    all_preserved = True
    for direction_name, direction in directions.items():
        for magnitude in stencil_magnitudes:
            margin = fdp.topology_margin(
                positions_ang, direction=direction, delta_ang=magnitude,
                cutoff_ang=cutoff_ang, lattice_vectors_ang=lattice_ang, method="central",
            )
            record = margin.to_dict()
            preserved = bool(record["topology_preserved"])
            all_preserved &= preserved
            cases.append({
                "direction": direction_name,
                "delta_ang": magnitude,
                "topology_sha256": input_signature_sha256(record),
                **record,
            })

    report = {
        "schema": "delta_translation_topology_preflight_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prod_sz_central": str(PROD_CENTRAL.relative_to(REPO_ROOT)),
        "cutoff_ang": cutoff_ang,
        "cutoff_mode": "per_atom_radius_sum (SIESTA PAO neighbour rule, rc_C + rc_C)",
        "delta_ladder_ang": list(ladder),
        "stencil_magnitudes_ang": stencil_magnitudes,
        "cases": cases,
        "verdict": "PASS" if all_preserved else "NO_GO",
        "claim_scope": (
            "prospective neighbour-topology check only; does not run SIESTA and "
            "does not itself certify Delta_PAO_cov or the translation gate"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "delta_translation_topology_preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"delta/translation topology preflight: {report['verdict']} -> {path}")
    return 0 if all_preserved else 1


if __name__ == "__main__":
    raise SystemExit(main())
