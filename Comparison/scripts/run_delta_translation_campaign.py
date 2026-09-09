#!/usr/bin/env python3
"""Run the 12 SIESTA jobs the pre-registered delta-plateau/translation_x campaign needs.

Reuses the exact PROD-SZ geometry template, DM.Tolerance override and SIESTA
launcher already used to produce ``PROD-SZ/C1_x/{m1,p1,m2,p2}`` (see
``run_displaced_projection_sentinel.py``), and the direction/displacement math
already used by the topology preflight (``fd_perturbation_space.py``). Every
parameter here must match ``delta_translation_campaign_preregistration_v1.json``;
this script does not decide the protocol, it only executes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

import fd_perturbation_space as fdp  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_displaced_projection_sentinel import PRODUCTION, _run_geometry  # noqa: E402

SIESTA = "/home/christian/bin/siesta"
PROD_SZ_ROOT = PROD_CENTRAL.parent  # .../basis_projection_sentinel/runs/PROD-SZ
PREREG = json.loads(
    (REPO_ROOT / "Comparison/results/epc/delta_translation_campaign_preregistration"
     / "delta_translation_campaign_preregistration_v1.json").read_text()
)


def _tag(delta_ang: float) -> str:
    magnitude = "d" + f"{abs(delta_ang):.6g}".replace(".", "p")
    return f"{magnitude}_{'plus' if delta_ang > 0 else 'minus'}"


def _plan() -> list[tuple[str, fdp.Direction, float, Path, str]]:
    xyz0 = sisl.get_sile(str(PRODUCTION / "equilibrium.TSHS")).read_geometry().xyz
    atom_count = xyz0.shape[0]
    directions = {
        "C1_x": fdp.one_hot(atom_count, 0, "x"),
        "translation_x": fdp.uniform_translation(atom_count, "x"),
    }
    jobs = []
    for delta in PREREG["runs"]["c1_x_new_ang"]:
        tag = _tag(delta)
        root = PROD_SZ_ROOT / "C1_x" / tag
        jobs.append(("C1_x", directions["C1_x"], delta, root, f"prod_c1_x_{tag}"))
    for delta in PREREG["runs"]["translation_x_new_ang"]:
        tag = _tag(delta)
        root = PROD_SZ_ROOT / "translation_x" / tag
        jobs.append(("translation_x", directions["translation_x"], delta, root, f"prod_translation_x_{tag}"))
    return xyz0, jobs


def main() -> int:
    if not PREREG.get("frozen"):
        raise RuntimeError("preregistration is not marked frozen; refusing to run")
    xyz0, jobs = _plan()
    for direction_name, direction, delta, root, label in jobs:
        xyz = np.asarray(fdp.displace(xyz0, direction, delta))
        print(f"[{direction_name}] delta={delta:+.6f} Ang -> {root}")
        _run_geometry(root, label, xyz, large_basis=False, siesta=SIESTA)
        print(f"  done: {root / (label + '.TSHS')}")
    print(f"campaign complete: {len(jobs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
