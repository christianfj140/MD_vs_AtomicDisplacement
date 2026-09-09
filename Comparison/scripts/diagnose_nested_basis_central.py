#!/usr/bin/env python3
"""Read-only central diagnostics for the non-converged nested PAO ladder."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from certify_epc_energy_zero import vacuum_level  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_ladder import BASIS_ORDER, COMPARE, IDENTITY_ROOT, _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/nested_basis_ladder/central_diagnostics.json"


def _last_total_energy(text: str) -> float:
    matches = re.findall(r"^siesta:\s+Etot\s+=\s+([-+0-9.Ee]+)", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError("SIESTA Etot missing")
    return float(matches[-1])


def _density(run_dir: Path, label: str, hamiltonian, atom, shape):
    import sisl

    geometry = _basis_geometry(hamiltonian.geometry, run_dir / "C.ion.xml")
    dm = sisl.get_sile(str(run_dir / f"{label}.DM")).read_density_matrix(geometry=geometry)
    grid = sisl.Grid(shape, lattice=geometry.lattice)
    dm.density(grid)
    return np.asarray(grid.grid)


def main() -> int:
    import sisl
    from nested_basis_identity_preflight import _selector

    _, h_a, _, _, atom_a = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / "prod_central.VT")).read_grid().shape)
    data = {}
    for basis in BASIS_ORDER:
        run = IDENTITY_ROOT / basis
        label, h, fermi, vacuum, atom = _matrix_data(run)
        selector, _ = _selector(atom_a, atom, h_a.geometry.na)
        density = _density(run, label, h, atom, shape)
        vt = np.asarray(sisl.get_sile(str(run / f"{label}.VT")).read_grid().grid)
        vh = np.asarray(sisl.get_sile(str(run / f"{label}.VH")).read_grid().grid)
        k_rows, k0 = [], {}
        for ik, k in enumerate(K_POINTS):
            s_b = np.asarray(h.Sk(k=k, format="array"))
            h_abs = np.asarray(h.Hk(k=k, format="array")) + fermi * s_b
            s_a = np.asarray(h_a.Sk(k=k, format="array"))
            k0[ik] = selector.T @ (h_abs - vacuum * s_b) @ selector
            eig = np.asarray(h.eigh(k=k))
            near = np.sort(eig[np.argsort(np.abs(eig))[: min(8, len(eig))]])
            k_rows.append({"k_reduced": list(k), "condition_S_B": float(np.linalg.cond(s_b)), "eight_bands_closest_to_EF_ev": near.tolist(), "K0_prod_subblock_rms_ev": _rms(k0[ik], s_a)[0]})
        data[basis] = {"label": label, "run": run, "h": h, "density": density, "vt": vt, "vh": vh, "vacuum": vacuum, "k0": k0}
        data[basis]["report"] = {
            "basis": basis, "total_energy_ev": _last_total_energy((run / "RUN.out").read_text(encoding="utf-8", errors="replace")),
            "fermi_ev": fermi, "vacuum_level_ev": vacuum, "k_results": k_rows,
            "dm_sha256": file_sha256(run / f"{label}.DM"), "vt_sha256": file_sha256(run / f"{label}.VT"),
        }

    comparisons = []
    for name, (upper, lower, _) in COMPARE.items():
        high, low = data[upper], data[lower]
        rho_diff = high["density"] - low["density"]
        vt_diff = (high["vt"] - high["vacuum"]) - (low["vt"] - low["vacuum"])
        vh_diff = (high["vh"] - vacuum_level(high["vh"])[0]) - (low["vh"] - vacuum_level(low["vh"])[0])
        row = {
            "comparison": name, "upper": upper, "lower": lower,
            "delta_total_energy_ev": high["report"]["total_energy_ev"] - low["report"]["total_energy_ev"],
            "density_rms": float(np.sqrt(np.mean(rho_diff**2))),
            "density_relative_l2": float(np.linalg.norm(rho_diff) / np.linalg.norm(high["density"])),
            "aligned_VT_rms_ev": float(np.sqrt(np.mean(vt_diff**2))),
            "aligned_VH_rms_ev": float(np.sqrt(np.mean(vh_diff**2))), "k_results": [],
        }
        for ik, k in enumerate(K_POINTS):
            s_a = np.asarray(h_a.Sk(k=k, format="array"))
            rms, maximum = _rms(high["k0"][ik] - low["k0"][ik], s_a)
            row["k_results"].append({"k_reduced": list(k), "K0_difference_rms_ev": rms, "K0_difference_max_ev": maximum})
        comparisons.append(row)
    report = {
        "schema": "nested_basis_central_diagnostics_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY", "trigger": "reference_basis_convergence_covariant=NO_GO",
        "bases": [data[basis]["report"] for basis in BASIS_ORDER], "comparisons": comparisons,
        "claim_ceiling": "descriptive diagnosis; no gate promotion and no Delta_out attribution",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"nested basis central diagnosis -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
