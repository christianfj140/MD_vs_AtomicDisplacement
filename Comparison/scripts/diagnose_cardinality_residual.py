#!/usr/bin/env python3
"""Free diagnostics for the covariant N-DZP -> N-TZP C1:x residual."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts")]

from cross_basis_projection_preflight import K_POINTS  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL, _radial_error, _signature  # noqa: E402

RUNS = REPO_ROOT / "Comparison/results/epc/nested_basis_identity_preflight/runs"
COVARIANT = REPO_ROOT / "Comparison/results/epc/reference_basis_convergence_covariant"
OUTPUT = COVARIANT / "cardinality_residual_diagnostics.json"


def _orbital_metadata(atom, atom_count: int) -> list[dict]:
    return [
        {"index": ia * atom.no + io, "atom": ia, "name": orbital.name(), "n": int(orbital.n),
         "l": int(orbital.l), "m": int(orbital.m), "zeta": int(orbital.zeta)}
        for ia in range(atom_count) for io, orbital in enumerate(atom.orbitals)
    ]


def _block_report(matrix: np.ndarray, orbitals: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[complex]] = {}
    for i, left in enumerate(orbitals):
        for j, right in enumerate(orbitals):
            angular = "ss" if left["l"] == right["l"] == 0 else "pp" if left["l"] == right["l"] == 1 else "sp"
            locality = "on_site" if left["atom"] == right["atom"] else "interatomic"
            groups.setdefault((angular, locality), []).append(matrix[i, j])
    total = float(np.linalg.norm(matrix) ** 2)
    return [
        {"angular_block": angular, "locality": locality, "frobenius_ev_per_ang": float(np.linalg.norm(values)),
         "squared_norm_fraction": float(np.linalg.norm(values) ** 2 / max(total, np.finfo(float).tiny)),
         "max_abs_ev_per_ang": float(np.max(np.abs(values)))}
        for (angular, locality), values in sorted(groups.items())
    ]


def _shared_radials(dzp, tzp) -> list[dict]:
    tz = {_signature(orbital): orbital for orbital in tzp.orbitals}
    return [
        {"orbital": orbital.name(), "signature": list(_signature(orbital)),
         "relative_error": _radial_error(orbital, tz[_signature(orbital)]),
         "dzp_cutoff_ang": float(orbital.R), "tzp_cutoff_ang": float(tz[_signature(orbital)].R)}
        for orbital in dzp.orbitals
    ]


def main() -> int:
    import sisl

    prod_h = sisl.get_sile(str(PROD_CENTRAL / "prod_central.TSHS")).read_hamiltonian()
    prod_atom = sisl.get_sile(str(PROD_CENTRAL / "C.ion.xml")).read_basis()
    dzp = sisl.get_sile(str(RUNS / "n_dzp/C.ion.xml")).read_basis()
    tzp = sisl.get_sile(str(RUNS / "n_tzp/C.ion.xml")).read_basis()
    low = np.load(COVARIANT / "n_dzp_C1_x_covariant.npz")
    high = np.load(COVARIANT / "n_tzp_C1_x_covariant.npz")
    orbitals = _orbital_metadata(prod_atom, prod_h.geometry.na)
    rows = []
    for ik, k in enumerate(K_POINTS):
        delta = high[f"Delta_covariant_k{ik}"] - low[f"Delta_covariant_k{ik}"]
        rows.append({"k_reduced": list(k), "raw_matrix_blocks": _block_report(delta, orbitals),
                     "largest_elements": sorted(
                         ({"row": orbitals[i], "column": orbitals[j], "abs_ev_per_ang": float(abs(delta[i, j]))}
                          for i in range(len(orbitals)) for j in range(len(orbitals))),
                         key=lambda item: item["abs_ev_per_ang"], reverse=True)[:8]})
    shared = _shared_radials(dzp, tzp)
    report = {
        "schema": "cardinality_residual_diagnostics_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY", "comparison": "N-TZP minus N-DZP, covariant, C1:x",
        "orbital_ordering_source": "PROD-SZ C.ion.xml; no p-axis ordering assumed", "production_orbitals": orbitals,
        "shared_radial_prefix": shared, "shared_radial_max_relative_error": max(row["relative_error"] for row in shared),
        "k_results": rows,
        "interpretation_limit": "Raw matrix block norms localize the residual but are not a causal or metric-whitened decomposition.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"cardinality residual diagnosis -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
