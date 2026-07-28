#!/usr/bin/env python
"""Backfill relative_frobenius into cross-structure / mixing sweep summaries.

``_record`` only stores relative_frobenius when the launch result carried one, so
campaigns whose metric came back through a path that dropped it ended up with MAE-only
records and no Frobenius curve in the UI. The value was never lost: every run writes it
to ``common_metrics/summary/common_method_metrics.csv``, so this recovers it by reading
that CSV per run root. Nothing is recomputed and no model is re-run.

Only records missing the field are touched; existing values are left alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# Column names differ between the per-run CSV (``*_mean``) and the aggregated one.
FROBENIUS_COLUMNS = ("relative_frobenius", "relative_frobenius_mean")


def frobenius_by_model(run_root: str | None) -> dict[str, float]:
    """model -> relative_frobenius read from a run's common metrics CSV."""
    if not run_root:
        return {}
    path = Path(str(run_root)) / "common_metrics" / "summary" / "common_method_metrics.csv"
    if not path.is_file():
        return {}
    found: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = str(row.get("method") or row.get("model") or "").lower()
            if not model:
                continue
            for column in FROBENIUS_COLUMNS:
                value = row.get(column)
                if value not in (None, ""):
                    try:
                        found[model] = float(value)
                    except ValueError:
                        pass
                    break
    return found


def run_roots_for_permutation(perm: dict[str, Any]) -> dict[str, str]:
    """model -> run_root. ``model_launches`` appears when models ran as separate stages."""
    launches = perm.get("model_launches")
    if isinstance(launches, dict) and launches:
        roots = {}
        for model, launch in launches.items():
            root = ((launch or {}).get("runner_status") or {}).get("run_root")
            if root:
                roots[str(model).lower()] = str(root)
        return roots
    root = ((perm.get("launch") or {}).get("runner_status") or {}).get("run_root")
    return {"*": str(root)} if root else {}


def backfill_summary(path: Path, *, apply: bool) -> tuple[int, int]:
    """Returns (filled, still_missing) for one summary JSON."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    records = summary.get("records") or []
    if not records:
        return 0, 0

    # payload_id -> {model: frobenius}, built once per summary.
    by_payload: dict[str, dict[str, float]] = {}
    for perm in summary.get("permutations") or []:
        payload_id = str(perm.get("payload_id") or "")
        merged: dict[str, float] = {}
        for model, root in run_roots_for_permutation(perm).items():
            values = frobenius_by_model(root)
            if model == "*":
                merged.update(values)
            elif model in values:
                merged[model] = values[model]
        if merged:
            by_payload[payload_id] = merged

    filled = missing = 0
    for record in records:
        if record.get("relative_frobenius") is not None:
            continue
        value = by_payload.get(str(record.get("payload_id")), {}).get(
            str(record.get("model")).lower()
        )
        if value is None:
            missing += 1
            continue
        record["relative_frobenius"] = float(value)
        filled += 1

    if filled and apply:
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return filled, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="*",
        help="Summary JSONs. Default: every cross-structure sweep summary under Comparison/results.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write the files (default is a dry run)."
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.summaries] or sorted(
        REPO_ROOT.glob("Comparison/results/**/cross_structure_sweep_summary.json")
    ) + sorted(REPO_ROOT.glob("Comparison/results/ml_vs_siesta_cross_structure_*/**/summary.json"))

    total_filled = total_missing = 0
    for path in paths:
        try:
            filled, missing = backfill_summary(path, apply=args.apply)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  !! {path}: {exc}")
            continue
        if filled or missing:
            rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
            print(f"  {'+' if filled else ' '}{filled:>4} rellenados, {missing:>4} sin CSV  {rel}")
        total_filled += filled
        total_missing += missing

    verb = "rellenados" if args.apply else "rellenables (dry-run)"
    print(f"\nTOTAL: {total_filled} {verb}, {total_missing} sin dato en CSV")
    if not args.apply and total_filled:
        print("Vuelve a lanzarlo con --apply para escribirlo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
