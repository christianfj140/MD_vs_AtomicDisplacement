#!/usr/bin/env python3
"""Build tiny mixing endpoint records from existing pure W90/5x5 metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SIZES = [20, 30, 50, 60, 80, 90, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
W90_12_SIZES = {30, 50, 90, 150, 400, 600}
W90_20_SIZES = {20, 60, 80, 100, 200, 300, 500, 800}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tar_text(archive: Path, member: str) -> str:
    return subprocess.check_output(
        ["tar", "--zstd", "-xOf", str(archive), member],
        cwd=REPO_ROOT,
        text=True,
    )


def size_from_config(config_id: str) -> int | None:
    match = re.search(r"N(\d+)", config_id)
    return int(match.group(1)) if match else None


def records_from_normalized(
    rows: list[dict[str, Any]],
    *,
    sizes: set[int],
    modes: tuple[str, ...],
    ratio: float,
    source: str,
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        size = size_from_config(str(row.get("config_id") or ""))
        if size not in sizes:
            continue
        for mode in modes:
            records.append(
                {
                    "size": size,
                    "mode": mode,
                    "ratio": ratio,
                    "seed": 0,
                    "total_size": size,
                    "model": row["model"],
                    "h_mae_eV": float(row["h_mae_eV_mean"]),
                    "relative_frobenius": row.get("relative_frobenius_mean"),
                    "split_policy": "blocked_stratified_gap",
                    "training_weighting_policy": "legacy_elementwise",
                    "reconstructed": True,
                    "reconstructed_source": source,
                }
            )
    return records


def metric_from_kpoint_csv(text: str) -> tuple[float, float | None]:
    rows = [row for row in csv.DictReader(text.splitlines()) if row.get("row_type") == "weighted_sample"]
    if not rows:
        raise RuntimeError("kpoint_matrix_metrics.csv has no weighted_sample rows")
    mae = sum(float(row["h_mae_eV"]) for row in rows) / len(rows)
    rel_values = [float(row["relative_frobenius"]) for row in rows if row.get("relative_frobenius")]
    rel = sum(rel_values) / len(rel_values) if rel_values else None
    return mae, rel


def w90_20_records(archive: Path) -> list[dict[str, Any]]:
    base = (
        "graphene_w90_snapshot_scaling_20_1100_600epochs_1train_derivatives/"
        "graphene_w90_snapshot_scaling_20_1100_600epochs_1train_derivatives/sweep"
    )
    records = []
    for size in sorted(W90_20_SIZES):
        for model, config_prefix, metrics_subdir in (
            ("graph2mat", "G2M-T1000-03-anchor", "graph2mat/eval_input"),
            ("deeph", "DH-T1000-04-anchor", "deeph/eval"),
        ):
            member = (
                f"{base}/{model}/graphene_w90_scale_iid{size}/"
                f"{config_prefix}-N{size}/metrics/{metrics_subdir}/metrics/kpoint_matrix_metrics.csv"
            )
            mae, rel = metric_from_kpoint_csv(tar_text(archive, member))
            for mode in ("add", "replace"):
                records.append(
                    {
                        "size": size,
                        "mode": mode,
                        "ratio": 0.0,
                        "seed": 0,
                        "total_size": size,
                        "model": model,
                        "h_mae_eV": mae,
                        "relative_frobenius": rel,
                        "split_policy": "blocked_stratified_gap",
                        "training_weighting_policy": "legacy_elementwise",
                        "reconstructed": True,
                        "reconstructed_source": str(archive),
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    w90_12 = REPO_ROOT / "Comparison/results_archived/results/graphene_w90_snapshot_scaling_12_1300_600epochs_1train_derivatives.tar.zst"
    w90_20 = REPO_ROOT / "Comparison/results_archived/results/graphene_w90_snapshot_scaling_20_1100_600epochs_1train_derivatives.tar.zst"
    g5 = REPO_ROOT / (
        "Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready/"
        "graphene_5x5_snapshot_scaling_complete_paper_ready/summary/ranking/normalized_run_metrics.json"
    )

    w90_12_member = (
        "graphene_w90_snapshot_scaling_12_1300_600epochs_1train_derivatives/"
        "graphene_w90_snapshot_scaling_12_1300_600epochs_1train_derivatives/"
        "summary/ranking/normalized_run_metrics.json"
    )
    records = []
    records.extend(
        records_from_normalized(
            json.loads(tar_text(w90_12, w90_12_member))["rows"],
            sizes=W90_12_SIZES,
            modes=("add", "replace"),
            ratio=0.0,
            source=str(w90_12),
        )
    )
    records.extend(w90_20_records(w90_20))
    records.extend(
        records_from_normalized(
            read_json(g5)["rows"],
            sizes=set(SIZES),
            modes=("replace",),
            ratio=1.0,
            source=str(g5),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "ml_vs_siesta_reconstructed_mixing_records_v1",
                "records": sorted(
                    records,
                    key=lambda r: (int(r["size"]), str(r["mode"]), float(r["ratio"]), str(r["model"])),
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
