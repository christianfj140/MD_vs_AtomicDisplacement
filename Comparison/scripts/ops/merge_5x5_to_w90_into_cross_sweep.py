#!/usr/bin/env python
"""Merge the 5x5->w90 campaign into the main cross-structure sweep summary.

The UI reads fixed roots (pipeline_ui: CROSS_TESTING_SWEEP_OUTPUT_ROOT), so results
written to ml_vs_siesta_cross_structure_5x5_to_w90 are invisible in the Cross testing
tab. The 5x5->w90 records already in the main summary are orphans: their run_root is
null, their artifacts are gone, and they carry no relative_frobenius, so nothing can
recover it for them. This replaces those orphans with the new campaign's records, which
have live artifacts, three real seeds, and Frobenius.

The w90->5x5 direction is never touched.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_SUMMARY = (
    REPO_ROOT / "Comparison/results/ml_vs_siesta_cross_structure_sweep/cross_structure_sweep_summary.json"
)
NEW_CAMPAIGN = REPO_ROOT / "Comparison/results/ml_vs_siesta_cross_structure_5x5_to_w90"


def is_5x5_to_w90(item: dict[str, Any]) -> bool:
    """Direction test used by the UI: 5x5 source and w90 target."""
    return "5x5" in str(item.get("source_id") or "") and "w90" in str(item.get("target_id") or "")


def load_new_campaign() -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    permutations: list[dict] = []
    for summary_path in sorted(NEW_CAMPAIGN.glob("seed_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        seed = int(str(summary_path.parent.name).rsplit("_", 1)[-1])
        for record in summary.get("records") or []:
            record = dict(record)
            record.setdefault("seed", seed)
            records.append(record)
        permutations.extend(summary.get("permutations") or [])
    return records, permutations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the file (default: dry run).")
    args = parser.parse_args()

    summary = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    new_records, _ = load_new_campaign()
    if not new_records:
        print("La campana 5x5->w90 no tiene registros todavia; nada que fusionar.")
        return 1

    kept_records = [r for r in (summary.get("records") or []) if not is_5x5_to_w90(r)]
    dropped = len(summary.get("records") or []) - len(kept_records)

    # Records only. The UI builds its curves from records, but the derivative launcher
    # (launch_ui_real_metrics_derivatives._cross_cases) walks *permutations* and resolves
    # Graph2Mat artifacts from each launch.runner_status.run_root. Under the
    # deeph_then_graph2mat schedule that run_root is the DeepH stage's, so injecting these
    # permutations makes the whole derivative pipeline die with
    # "Incomplete Graph2Mat artifacts".
    summary["records"] = kept_records + new_records
    summary["merged_from"] = {
        "campaign": str(NEW_CAMPAIGN.relative_to(REPO_ROOT)),
        "direction": "5x5_to_w90",
        "note": (
            "Checkpoints entrenados en graphene 5x5 PRISTINO (50 atomos), evaluados en "
            "graphene w90 (2 atomos). No es un modelo entrenado sobre la vacante."
        ),
    }

    frob = sum(1 for r in summary["records"] if r.get("relative_frobenius") is not None)
    print(f"  reemplazados: {dropped} registros huerfanos 5x5->w90 (permutaciones intactas)")
    print(f"  anadidos:     {len(new_records)} registros ({len({r.get('seed') for r in new_records})} seeds)")
    print(f"  resultado:    {len(summary['records'])} registros, {frob} con relative_frobenius")

    if not args.apply:
        print("\nDry run. Relanza con --apply para escribirlo.")
        return 0

    backup = MAIN_SUMMARY.with_suffix(".json.bak")
    shutil.copy2(MAIN_SUMMARY, backup)
    MAIN_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nEscrito. Copia previa en {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
