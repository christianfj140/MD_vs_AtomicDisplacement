#!/usr/bin/env python
"""Publish the vacancy->w90 campaign into the Cross testing vacancy tab.

The UI reads fixed roots (pipeline_ui: CROSS_TESTING_VACANCY_OUTPUT_ROOT), so a campaign
written anywhere else is invisible. Pointing the campaign's --output-root straight at the
vacancy root is not an option either: run_cross_structure_sweep writes
cross_structure_sweep_summary.json there, which would clobber the existing 114-record
vacancy campaign.

So the campaign runs in its own directory and this merges its *records* in. Records only:
the derivative launcher walks permutations and resolves Graph2Mat artifacts from
launch.runner_status.run_root, and injecting permutations from a differently-staged
campaign breaks it with "Incomplete Graph2Mat artifacts".
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VACANCY_SUMMARY = (
    REPO_ROOT
    / "Comparison/results/ml_vs_siesta_cross_structure_vacancy/cross_structure_sweep_summary.json"
)
CAMPAIGN = REPO_ROOT / "Comparison/results/ml_vs_siesta_cross_structure_vacancy_to_w90"


def is_vacancy_to_w90(record: dict) -> bool:
    return "vacancy" in str(record.get("source_id") or "") and "w90" in str(
        record.get("target_id") or ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the file (default: dry run).")
    args = parser.parse_args()

    summary_path = CAMPAIGN / "summary.json"
    if not summary_path.is_file():
        print(f"Sin resultados todavia en {summary_path}; nada que publicar.")
        return 1
    new_records = json.loads(summary_path.read_text(encoding="utf-8")).get("records") or []
    if not new_records:
        print("La campana no produjo registros; no se publica nada.")
        return 1

    summary = json.loads(VACANCY_SUMMARY.read_text(encoding="utf-8"))
    kept = [r for r in (summary.get("records") or []) if not is_vacancy_to_w90(r)]
    dropped = len(summary.get("records") or []) - len(kept)
    summary["records"] = kept + new_records
    summary.setdefault("merged_campaigns", []).append(
        {
            "campaign": str(CAMPAIGN.relative_to(REPO_ROOT)),
            "direction": "vacancy_to_w90",
            "note": (
                "Modelos entrenados SOBRE la monovacante (49 atomos, MD propia) y evaluados en "
                "graphene w90 (2 atomos). Distinto de las curvas *->vacancy, donde la vacante es el test."
            ),
        }
    )
    frob = sum(1 for r in summary["records"] if r.get("relative_frobenius") is not None)
    print(f"  reemplazados: {dropped} registros vacancy->w90 previos")
    print(f"  anadidos:     {len(new_records)}")
    print(f"  resultado:    {len(summary['records'])} registros, {frob} con relative_frobenius")

    if not args.apply:
        print("\nDry run. Relanza con --apply para escribirlo.")
        return 0
    shutil.copy2(VACANCY_SUMMARY, VACANCY_SUMMARY.with_suffix(".json.bak"))
    VACANCY_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nPublicado en la pestana Cross testing vacancy (backup .bak creado).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
