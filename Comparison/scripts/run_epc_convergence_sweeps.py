#!/usr/bin/env python3
"""Run the missing graphene SIESTA numerical-convergence campaigns."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from run_epc_siesta_reference import run_campaign  # noqa: E402

SOURCE_FDF = REPO_ROOT / "materials/graphene/RUN.fdf"
ROOT = REPO_ROOT / "Comparison/results/epc/convergence"

CAMPAIGNS = {
    "gauge_delta": ({}, [0.0025, 0.005, 0.01, 0.02, 0.04]),
    "gauge_e2g": ({}, [0.005, 0.01, 0.02]),
    **{f"scf_{value:g}": ({"scf": value}, [0.005, 0.01, 0.02]) for value in (1e-4, 3e-5, 1e-5, 3e-6)},
    **{f"mesh_{value}": ({"mesh": value}, [0.005, 0.01, 0.02]) for value in (400, 600, 800, 1000)},
    **{f"kgrid_{value}": ({"kgrid": value}, [0.005, 0.01, 0.02]) for value in (16, 20, 24, 30)},
}


def materialize_variant(source: str, *, mesh: int | None = None,
                        scf: float | None = None, kgrid: int | None = None) -> str:
    text = source
    if mesh is not None:
        text, count = re.subn(r"(?mi)^MeshCutoff\s+\S+\s+Ry\s*$", f"MeshCutoff             {mesh}.0 Ry", text)
        if count != 1:
            raise ValueError("expected exactly one MeshCutoff directive")
    if scf is not None:
        text, count = re.subn(r"(?mi)^DM\.Tolerance\s+\S+\s*$", f"DM.Tolerance           {scf:.1e}", text)
        if count != 1:
            raise ValueError("expected exactly one DM.Tolerance directive")
    if kgrid is not None:
        replacement = (
            "%block kgrid_Monkhorst_Pack\n"
            f"  {kgrid}   0   0  0.0\n"
            f"   0  {kgrid}   0  0.0\n"
            "   0   0   1  0.0\n"
            "%endblock Kgrid_Monkhorst_Pack"
        )
        text, count = re.subn(
            r"(?ims)^%block kgrid_Monkhorst_Pack\s*$.*?^%endblock Kgrid_Monkhorst_Pack\s*$",
            replacement,
            text,
        )
        if count != 1:
            raise ValueError("expected exactly one kgrid_Monkhorst_Pack block")
    return text.rstrip() + "\n\nSaveTotalPotential         true\nSaveElectrostaticPotential true\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=CAMPAIGNS, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    selected = args.only or list(CAMPAIGNS)
    source = SOURCE_FDF.read_text(encoding="utf-8")
    summary: dict[str, object] = {"schema": "epc_convergence_sweeps_v1", "campaigns": {}}
    for name in selected:
        changes, deltas = CAMPAIGNS[name]
        root = ROOT / name
        fdf = root / "input" / "RUN.fdf"
        fdf.parent.mkdir(parents=True, exist_ok=True)
        fdf.write_text(materialize_variant(source, **changes), encoding="utf-8")
        print(f"[EPC-CONVERGENCE] {name}: {changes or 'baseline'}, deltas={deltas}", flush=True)
        directions = None
        if name == "gauge_e2g":
            import preregister_graphene_gamma_epc as prereg
            import run_graphene_gamma_epc_paths as gamma_paths

            protocol = prereg.load_protocol(prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME)
            directions = list(gamma_paths.protocol_directions(protocol).values())
        manifest = run_campaign(
            material_fdf=fdf,
            basis_dir=SOURCE_FDF.parent / "basis",
            pseudo_dir=SOURCE_FDF.parent / "pseudos",
            output_root=root,
            delta_ang_values=deltas,
            directions=directions,
            dry_run=args.dry_run,
        )
        summary["campaigns"][name] = {
            "changes": changes,
            "deltas": deltas,
            "manifest": str(root / "epc_siesta_reference_manifest.json"),
            "runs_failed": manifest["runs_failed"],
            "runs_certified": manifest["runs_certified"],
        }
        (ROOT / "sweep_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if manifest["runs_failed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
