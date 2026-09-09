#!/usr/bin/env python3
"""Run the 12 SIESTA jobs the pre-registered direct E2g_y campaign needs.

Reuses the exact PROD-SZ geometry template, DM.Tolerance override and SIESTA
launcher already used for delta_translation (see run_delta_translation_
campaign.py), and fd_perturbation_space.collective() for the modal direction:
C1 = +Q/sqrt(2) y-hat, C2 = -Q/sqrt(2) y-hat.
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
PROD_SZ_ROOT = PROD_CENTRAL.parent
PREREG = json.loads(
    (REPO_ROOT / "Comparison/results/epc/e2g_y_campaign_preregistration"
     / "e2g_y_campaign_preregistration_v1.json").read_text()
)


def _tag(delta_ang: float) -> str:
    magnitude = "d" + f"{abs(delta_ang):.6g}".replace(".", "p")
    return f"{magnitude}_{'plus' if delta_ang > 0 else 'minus'}"


def main() -> int:
    if not PREREG.get("frozen"):
        raise RuntimeError("preregistration is not frozen; refusing to run")

    xyz0 = sisl.get_sile(str(PRODUCTION / "equilibrium.TSHS")).read_geometry().xyz
    direction = fdp.collective([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], name="e2g_y")

    for q in PREREG["new_q_ang"]:
        xyz = np.asarray(fdp.displace(xyz0, direction, q))
        tag = _tag(q)
        root = PROD_SZ_ROOT / "e2g_y" / tag
        label = f"prod_e2g_y_{tag}"
        print(f"[e2g_y] Q={q:+.6f} Ang -> {root}")
        _run_geometry(root, label, xyz, large_basis=False, siesta=SIESTA)
        print(f"  done: {root / (label + '.TSHS')}")
    print(f"campaign complete: {len(PREREG['new_q_ang'])} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
