#!/usr/bin/env python3
"""Aggregate fair Graph2Mat-vs-DeepH k-point benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from deeph_fair_utils import read_json, run_git_commit, write_csv_rows, write_json


METRICS = [
    "h_mae_eV",
    "h_rmse_eV",
    "relative_frobenius",
    "global_rmse_eV",
    "fermi_window_rmse_eV",
    "frontier_window_rmse_eV",
    "dos_mae_500_fermi_window",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def summarize_method(method: str, result_dir: Path, metrics_root: Path | None = None) -> dict[str, Any]:
    metrics_root = metrics_root or result_dir / "metrics"
    matrix_rows = [row for row in read_rows(metrics_root / "kpoint_matrix_metrics.csv") if row.get("row_type") == "weighted_sample"]
    spectral_rows = read_rows(metrics_root / "kpoint_spectral_metrics.csv")
    dos_rows = read_rows(metrics_root / "kpoint_dos_metrics.csv")
    manifest = read_json(metrics_root / "manifest.json")
    summary: dict[str, Any] = {
        "method": method,
        "result_dir": str(result_dir),
        "metrics_root": str(metrics_root),
        "samples_compared": manifest.get("samples_compared"),
        "samples_failed": manifest.get("samples_failed"),
        "kpoint_metrics_enabled": manifest.get("kpoint_metrics_enabled"),
        "uses_reference_overlap_k": manifest.get("uses_reference_overlap_k"),
    }
    source_map = {
        "h_mae_eV": matrix_rows,
        "h_rmse_eV": matrix_rows,
        "relative_frobenius": matrix_rows,
        "global_rmse_eV": spectral_rows,
        "fermi_window_rmse_eV": spectral_rows,
        "frontier_window_rmse_eV": spectral_rows,
        "dos_mae_500_fermi_window": dos_rows,
    }
    for metric in METRICS:
        summary[f"{metric}_mean"] = mean([number(row.get(metric)) for row in source_map[metric]])
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_method("graph2mat", args.graph2mat_result_dir),
        summarize_method("deeph", args.deeph_eval_dir, args.deeph_eval_dir / "metrics"),
    ]
    write_csv_rows(args.output_dir / "aggregate_graph2mat_vs_deeph.csv", rows)
    sample_rows: list[dict[str, Any]] = []
    for method, result_dir in [("graph2mat", args.graph2mat_result_dir), ("deeph", args.deeph_eval_dir)]:
        metrics_root = result_dir / "metrics"
        for row in read_rows(metrics_root / "kpoint_matrix_metrics.csv"):
            if row.get("row_type") == "weighted_sample":
                sample_rows.append({"method": method, **row})
    write_csv_rows(args.output_dir / "per_sample_comparison.csv", sample_rows)
    worst = sorted(
        sample_rows,
        key=lambda row: number(row.get("h_rmse_eV")),
        reverse=True,
    )[:20]
    write_csv_rows(args.output_dir / "worst_samples.csv", worst)
    g2m = rows[0]
    deeph = rows[1]
    report = [
        "# Graph2Mat vs DeepH Diagnostic H-MAE Summary",
        "",
        "This report compares only metrics produced on the same repository k-point schema.",
        "It is a supporting/diagnostic summary, not the final paper-ready winner report.",
        "Robust scientific claims must come from final_statistics, gate_check and the preregistered final_evaluation metric.",
        "",
        "## Aggregate Metrics",
        "",
        "| method | samples | H(k) MAE meV | H(k) RMSE meV | rel Frobenius | global RMSE eV | Fermi RMSE eV | DOS MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {method} | {samples} | {h_mae:.3f} | {h_rmse:.3f} | {rel:.6f} | {global_rmse:.6f} | {fermi:.6f} | {dos:.6f} |".format(
                method=row["method"],
                samples=row.get("samples_compared"),
                h_mae=1000 * number(row.get("h_mae_eV_mean")),
                h_rmse=1000 * number(row.get("h_rmse_eV_mean")),
                rel=number(row.get("relative_frobenius_mean")),
                global_rmse=number(row.get("global_rmse_eV_mean")),
                fermi=number(row.get("fermi_window_rmse_eV_mean")),
                dos=number(row.get("dos_mae_500_fermi_window_mean")),
            )
        )
    verdict = "inconclusive"
    if math.isfinite(number(g2m.get("h_mae_eV_mean"))) and math.isfinite(number(deeph.get("h_mae_eV_mean"))):
        verdict = "deeph_better_h_mae" if number(deeph["h_mae_eV_mean"]) < number(g2m["h_mae_eV_mean"]) else "graph2mat_better_h_mae"
    report.extend(
        [
            "",
            "## Verdict",
            "",
            f"Supporting H(k) MAE diagnostic comparison: `{verdict}`.",
            "",
            "Do not use this H-MAE diagnostic verdict as a final spectral-quality or winner claim.",
            "Do not use DeepH training loss or Graph2Mat training loss as the primary scientific comparison.",
        ]
    )
    (args.output_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "stage": "graph2mat_deeph_compare",
        "graph2mat_result_dir": str(args.graph2mat_result_dir.resolve()),
        "deeph_eval_dir": str(args.deeph_eval_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "pipeline_git": run_git_commit(Path.cwd()),
        "claim_scope": "diagnostic_supporting_h_mae_only",
        "robust_winner_claim_allowed": False,
        "final_claim_source": "g2m_deeph_final_stats + g2m_deeph_gate_check + g2m_deeph_report",
    }
    write_json(args.output_dir / "comparison_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph2mat-result-dir", type=Path, required=True)
    parser.add_argument("--deeph-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Comparison/results/graphene_w90_deeph_fair_benchmark"))
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(f"[DEEPh-FAIR] comparison written to {manifest['output_dir']}")


if __name__ == "__main__":
    main()
