#!/usr/bin/env python3
"""Certify a common vacuum-potential gauge for displaced graphene runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))
DEFAULT_ROOT = REPO_ROOT / "Comparison/results/epc/convergence/gauge_delta"

from certify_siesta_dhsdr import (  # noqa: E402
    align, contract_fc, difference_norms, load_campaign, norms, read_tshs,
)
from read_siesta_dhsdr import read_dhsdr  # noqa: E402
from artifact_signature import file_sha256  # noqa: E402


def vacuum_level(grid: np.ndarray, fraction: float = 0.2) -> tuple[float, float]:
    """Mean and standard deviation in the cell-centred vacuum slab."""
    profile = np.asarray(grid, dtype=float).mean(axis=(0, 1))
    width = max(1, int(round(len(profile) * fraction)))
    start = (len(profile) - width) // 2
    window = profile[start:start + width]
    return float(window.mean()), float(window.std())


def combine(*terms: tuple[float, dict]) -> dict:
    keys = set().union(*(field for _, field in terms))
    return {key: sum(scale * field.get(key, 0.0) for scale, field in terms) for key in keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    manifest_path = args.reference_root / "epc_siesta_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["run_id"]: row for row in manifest["rows"]}
    campaign = load_campaign(args.reference_root)

    import sisl

    levels: dict[str, dict[str, float | str]] = {}
    for run_id, row in rows.items():
        if row.get("branch") not in {"equilibrium", "fd_central"} or not row.get("certified"):
            continue
        run_dir = Path(row["run_dir"])
        label = row["system_label"]
        entry: dict[str, float | str] = {}
        for kind in ("VT", "VH"):
            path = run_dir / f"{label}.{kind}"
            if not path.is_file():
                raise FileNotFoundError(path)
            mean, std = vacuum_level(sisl.get_sile(str(path)).read_grid().grid)
            entry[f"{kind.lower()}_vacuum_ev"] = mean
            entry[f"{kind.lower()}_vacuum_std_ev"] = std
            entry[f"{kind.lower()}_path"] = str(path)
            entry[f"{kind.lower()}_sha256"] = file_sha256(path)
        entry["vt_minus_vh_vacuum_ev"] = float(entry["vt_vacuum_ev"]) - float(entry["vh_vacuum_ev"])
        levels[run_id] = entry

    equilibrium_id = manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id,
        fermi_ev=campaign.fermi_ev(equilibrium_id),
    )
    c0 = float(levels[equilibrium_id]["vt_vacuum_ev"])
    pairs = []
    aligned_fd_by_pair: dict[tuple[str, float], dict] = {}
    for pair in manifest["certification_set"]["fd_pairs"]:
        plus, minus = levels[pair["plus_run_id"]], levels[pair["minus_run_id"]]
        delta = float(pair["delta_ang"])
        dc = (float(plus["vt_vacuum_ev"]) - float(minus["vt_vacuum_ev"])) / (2.0 * delta)
        plus_tshs = read_tshs(
            campaign.run_dir(pair["plus_run_id"]), pair["plus_run_id"],
            fermi_ev=campaign.fermi_ev(pair["plus_run_id"]),
        )
        minus_tshs = read_tshs(
            campaign.run_dir(pair["minus_run_id"]), pair["minus_run_id"],
            fermi_ev=campaign.fermi_ev(pair["minus_run_id"]),
        )
        aligned_fd = combine(
            (1.0 / (2.0 * delta), plus_tshs.h_absolute),
            (-float(plus["vt_vacuum_ev"]) / (2.0 * delta), plus_tshs.overlap),
            (-1.0 / (2.0 * delta), minus_tshs.h_absolute),
            (float(minus["vt_vacuum_ev"]) / (2.0 * delta), minus_tshs.overlap),
        )
        direction = next(
            item for item in manifest["perturbation_space"]["directions"]
            if item["direction_name"] == pair["direction_name"]
        )
        dhsdr = read_dhsdr(campaign.dhsdr_path(delta))
        vectors = np.asarray(direction["vectors"], dtype=float)
        fc_h = contract_fc(dhsdr, vectors, "D_H")
        fc_s = contract_fc(dhsdr, vectors, "D_S")
        aligned_fc = combine((1.0, fc_h), (-c0, fc_s), (-dc, equilibrium.overlap))
        discrepancy = difference_norms(aligned_fc, aligned_fd)["frobenius"]
        signal = norms(align(aligned_fd)[1][0])["frobenius"]
        aligned_fd_by_pair[(pair["direction_name"], delta)] = aligned_fd
        pairs.append({
            "direction_name": pair["direction_name"],
            "delta_ang": delta,
            "plus_run_id": pair["plus_run_id"],
            "minus_run_id": pair["minus_run_id"],
            "dc_vacuum_ev_per_ang": dc,
            "aligned_fd_frobenius_ev_per_ang": signal,
            "aligned_fc_fd_discrepancy_ev_per_ang": discrepancy,
            "aligned_fc_fd_relative_discrepancy": discrepancy / signal if signal else None,
        })

    richardson = []
    for (direction, fine), fine_field in sorted(aligned_fd_by_pair.items()):
        coarse_field = aligned_fd_by_pair.get((direction, 2.0 * fine))
        if coarse_field is None:
            continue
        extrapolated = combine((4.0 / 3.0, fine_field), (-1.0 / 3.0, coarse_field))
        richardson.append({
            "direction_name": direction,
            "fine_delta_ang": fine,
            "coarse_delta_ang": 2.0 * fine,
            "scheme": "(4 D_h - D_2h) / 3 after H-c_vacuum*S alignment",
            "correction_frobenius_ev_per_ang": difference_norms(extrapolated, fine_field)["frobenius"],
        })

    flatness_limit_ev = 1e-4
    vt_vh_limit_ev = 1e-6
    passed = bool(levels) and all(
        float(row["vt_vacuum_std_ev"]) <= flatness_limit_ev
        and abs(float(row["vt_minus_vh_vacuum_ev"])) <= vt_vh_limit_ev
        for row in levels.values()
    )
    report = {
        "schema": "epc_energy_zero_alignment_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "NO_GO",
        "gauge": "cell-centred vacuum mean of SIESTA VT",
        "alignment_rule": "H_aligned(R) = H(R) - c_vacuum(R) S(R)",
        "claim_ceiling": "finite-PAO covariant response; this does not bound Delta_out",
        "thresholds": {
            "vacuum_profile_std_ev": flatness_limit_ev,
            "abs_vt_minus_vh_vacuum_ev": vt_vh_limit_ev,
        },
        "reference_manifest": str(manifest_path),
        "levels": levels,
        "directional_offsets": pairs,
        "richardson": richardson,
    }
    output = args.reference_root / "energy_zero_alignment.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"energy-zero gauge: {report['verdict']} -> {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
