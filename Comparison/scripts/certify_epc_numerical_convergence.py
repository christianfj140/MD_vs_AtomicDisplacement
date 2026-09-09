#!/usr/bin/env python3
"""Compare the executed SCF, MeshCutoff and k-grid derivative sweeps."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from certify_siesta_dhsdr import align, contract_fc, difference_norms, load_campaign, norms  # noqa: E402
from read_siesta_dhsdr import read_dhsdr  # noqa: E402

ROOT = REPO_ROOT / "Comparison/results/epc/convergence"
FAMILIES = {
    "scf": ("scf_1e-05", "scf_3e-06"),
    "mesh": ("mesh_800", "mesh_1000"),
    "kgrid": ("kgrid_24", "kgrid_30"),
}
RELATIVE_LIMIT = 0.01


def fields(name: str, delta: float = 0.005):
    campaign = load_campaign(ROOT / name)
    dhsdr = read_dhsdr(campaign.dhsdr_path(delta))
    return {
        (direction["direction_name"], kind): contract_fc(
            dhsdr, np.asarray(direction["vectors"], dtype=float), kind
        )
        for direction in campaign.manifest["perturbation_space"]["directions"]
        for kind in ("D_H", "D_S")
    }


def main() -> int:
    report = {
        "schema": "epc_numerical_convergence_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "relative_limit": RELATIVE_LIMIT,
        "families": {},
        "claim_ceiling": "finite-PAO matrix response; basis and Delta_out are not certified",
    }
    passed = True
    for family, (coarse_name, fine_name) in FAMILIES.items():
        coarse, fine = fields(coarse_name), fields(fine_name)
        signal_scales = {
            kind: max(
                norms(align(field)[1][0])["frobenius"]
                for (direction, field_kind), field in fine.items()
                if field_kind == kind and direction != "translation_x"
            )
            for kind in ("D_H", "D_S")
        }
        rows = []
        for key in sorted(fine):
            signal = norms(align(fine[key])[1][0])["frobenius"]
            error = difference_norms(coarse[key], fine[key])["frobenius"]
            null_direction = key[0] == "translation_x"
            denominator = signal_scales[key[1]] if null_direction else signal
            relative = error / denominator if denominator else None
            row_passed = relative is None or relative <= RELATIVE_LIMIT
            passed &= row_passed
            rows.append({
                "direction_name": key[0], "kind": key[1],
                "coarse": coarse_name, "fine": fine_name,
                "fine_frobenius": signal, "difference_frobenius": error,
                "relative_difference": relative, "null_direction": null_direction,
                "normalization_frobenius": denominator, "passed": row_passed,
            })
        report["families"][family] = {"passed": all(row["passed"] for row in rows), "rows": rows}
    report["verdict"] = "PASS" if passed else "NO_GO"
    output = ROOT / "numerical_convergence.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"numerical convergence: {report['verdict']} -> {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
