#!/usr/bin/env python3
"""Build a compact cross-evaluation CSV from archived Hamiltonian metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_RESULTS = REPO_ROOT / "Comparison" / "results"


def latest_run(root: Path) -> Path | None:
    candidates = sorted(root.glob("dataset_*/run_*/*"), key=lambda path: path.stat().st_mtime)
    result_dirs = [path.parent for path in candidates if path.name == "manifest.json"]
    return result_dirs[-1] if result_dirs else None


def load_metrics(result_dir: Path | None) -> dict[str, Any] | None:
    if result_dir is None:
        return None
    manifest_path = result_dir / "metrics" / "manifest.json"
    if not manifest_path.exists():
        manifest_path = result_dir / "eigenvalues" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"No encontre manifest de metricas en {result_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def mean_metric(manifest: dict[str, Any], section: str, key: str) -> float | None:
    value = manifest.get("summary", {}).get(section, {}).get(key, {}).get("mean")
    return None if value is None else float(value)


def format_cell(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return ""
    sparse_mae = mean_metric(manifest, "sparse", "mae_ref_eV")
    sparse_rmse = mean_metric(manifest, "sparse", "rmse_ref_eV")
    fermi_rmse = mean_metric(manifest, "spectral", "fermi_window_rmse_eV")
    parts = []
    if sparse_mae is not None:
        parts.append(f"MAE={sparse_mae:.6g}")
    if sparse_rmse is not None:
        parts.append(f"RMSE={sparse_rmse:.6g}")
    if fermi_rmse is not None:
        parts.append(f"FermiRMSE={fermi_rmse:.6g}")
    return "; ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Modelo x Test CSV from comparison result directories."
    )
    parser.add_argument("--md-on-md", type=Path, default=None)
    parser.add_argument("--md-on-fc", type=Path, default=None)
    parser.add_argument("--md-on-mixed", type=Path, default=None)
    parser.add_argument("--fc-on-md", type=Path, default=None)
    parser.add_argument("--fc-on-fc", type=Path, default=None)
    parser.add_argument("--fc-on-mixed", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=COMPARISON_RESULTS / "comparison" / "metrics.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-incomplete-legacy",
        action="store_true",
        help="Allow missing cells for legacy exploratory tables. Strict mode is the default.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    md_on_md = args.md_on_md or latest_run(COMPARISON_RESULTS / "results_md")
    fc_on_fc = args.fc_on_fc or latest_run(COMPARISON_RESULTS / "results_atomdisp")
    paths = {
        "md_on_md": md_on_md,
        "md_on_fc": args.md_on_fc,
        "md_on_mixed": args.md_on_mixed,
        "fc_on_md": args.fc_on_md,
        "fc_on_fc": fc_on_fc,
        "fc_on_mixed": args.fc_on_mixed,
    }
    missing = sorted(name for name, path in paths.items() if path is None)
    if missing and not args.allow_incomplete_legacy:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Incomplete cross-evaluation grid. Pass --allow-incomplete-legacy for exploratory tables.",
                    "missing_cells": missing,
                },
                ensure_ascii=False,
            )
        )
        return 2
    matrix = {
        "Modelo_MD": {
            "Test_MD": load_metrics(md_on_md),
            "Test_FC": load_metrics(args.md_on_fc),
            "Test_mixto": load_metrics(args.md_on_mixed),
        },
        "Modelo_FC": {
            "Test_MD": load_metrics(args.fc_on_md),
            "Test_FC": load_metrics(fc_on_fc),
            "Test_mixto": load_metrics(args.fc_on_mixed),
        },
    }
    rows = [
        {
            "Modelo": model,
            "Test_MD": format_cell(cells["Test_MD"]),
            "Test_FC": format_cell(cells["Test_FC"]),
            "Test_mixto": format_cell(cells["Test_mixto"]),
        }
        for model, cells in matrix.items()
    ]
    if args.dry_run:
        print(json.dumps(rows, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Modelo", "Test_MD", "Test_FC", "Test_mixto"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Tabla cruzada escrita en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
