#!/usr/bin/env python3
"""Plot E2g derivative-matrix error against Graph2Mat training set size."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "Comparison/scripts/run_graphene_gamma_epc_paths.py"
OUTPUT = ROOT / "Comparison/results/epc/snapshot_scaling_20260908"
UI_OUTPUT = ROOT / "Comparison/results/ui_real_metrics_derivatives/epc/snapshot_scaling_20260908"
SIZES = (20, 50, 100, 200, 300, 400, 500, 600)
DIRECTIONS = ("e2g_bond_longitudinal", "e2g_bond_transverse")


def checkpoint(family: str, size: int) -> Path:
    if family == "w90":
        root = ROOT / "Comparison/results/ml_vs_siesta_cross_structure_vacancy/seed_1"
        pattern = (
            f"graphene__graphene_w90_scale_iid{size}__to__graphene_5x5_vacancy/"
            "training/**/checkpoints/best*.ckpt"
        )
    else:
        root = ROOT / "Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready"
        pattern = f"**/graphene_5x5_scale_iid{size}/**/checkpoints/best*.ckpt"
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {family}-{size} checkpoint, found {len(matches)}")
    return matches[0]


def evaluate(family: str, size: int, source: Path) -> Path:
    destination = OUTPUT / "runs" / f"{family}_{size}"
    summary = destination / "graphene_gamma_epc_summary.json"
    if summary.is_file():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(RUNNER), "--checkpoint", str(source),
             "--result-root", str(destination)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"{family}-{size} failed; see {destination / 'run.log'}")
    return destination


def matrix_metrics(run: Path) -> dict[str, float]:
    predictions, references = [], []
    for direction in DIRECTIONS:
        pred = np.load(run / "arrays" / f"raw_derivative__graph2mat_jvp__{direction}__d0p02.npz")["D_H"]
        ref = np.load(run / "arrays" / f"raw_derivative__siesta_reference__{direction}__d0p02.npz")["D_H"]
        predictions.append(pred.reshape(-1))
        references.append(ref.reshape(-1))
    pred, ref = np.concatenate(predictions), np.concatenate(references)
    support = (pred != 0) | (ref != 0)
    pred, ref = pred[support], ref[support]
    delta = pred - ref
    abs_delta = np.abs(delta)
    ref_support, pred_support = ref != 0, pred != 0
    true_support = np.count_nonzero(ref_support & pred_support)
    precision = true_support / max(np.count_nonzero(pred_support), 1)
    recall = true_support / max(np.count_nonzero(ref_support), 1)
    mae = float(np.mean(abs_delta))
    return {
        "dh_mae_union_eV_per_Ang": mae,
        "dh_rmse_union_eV_per_Ang": float(np.sqrt(np.mean(delta**2))),
        "dh_relative_frobenius_ref": float(np.linalg.norm(delta) / np.linalg.norm(ref)),
        "dh_relative_frobenius_union_robust": float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-15)),
        "dh_relative_l1_union_robust": float(np.sum(abs_delta) / max(np.sum(np.abs(ref)), 1e-15)),
        "dh_support_f1": 2 * precision * recall / max(precision + recall, 1e-15),
        "dh_false_zero_rate": float(np.count_nonzero(ref_support & ~pred_support) / max(np.count_nonzero(ref_support), 1)),
        "dh_false_nonzero_rate": float(np.count_nonzero(pred_support & ~ref_support) / max(np.count_nonzero(pred_support), 1)),
        "dh_pearson_union": float(np.corrcoef(pred, ref)[0, 1]),
        "dh_residual_mean_union_eV_per_Ang": float(np.mean(delta)),
        "dh_residual_std_union_eV_per_Ang": float(np.std(delta)),
        "dh_residual_median_union_eV_per_Ang": float(np.median(delta)),
        "dh_residual_bias_over_mae_union": float(abs(np.mean(delta)) / max(mae, 1e-15)),
        "dh_residual_abs_p90_union_eV_per_Ang": float(np.percentile(abs_delta, 90)),
        "dh_residual_abs_p95_union_eV_per_Ang": float(np.percentile(abs_delta, 95)),
        "dh_residual_abs_p99_union_eV_per_Ang": float(np.percentile(abs_delta, 99)),
    }


def g_metrics(run: Path, path: str = "graph2mat_frozen") -> dict[str, float]:
    reports = json.loads((run / "graphene_gamma_epc_g_blocks.json").read_text())["reports"]
    reference = {
        (row["mode_label"], row["k_label"]): np.asarray(row["g"]["values_real"])
        + 1j * np.asarray(row["g"]["values_imag"])
        for row in reports if row["path"] == "siesta_reference"
    }
    grouped = {}
    for row in reports:
        if row["path"] != path:
            continue
        values = np.asarray(row["g"]["values_real"]) + 1j * np.asarray(row["g"]["values_imag"])
        target = reference[(row["mode_label"], row["k_label"])]
        grouped.setdefault(row["k_label"], []).append(float(np.linalg.norm(values - target) / np.linalg.norm(target)))
    return {
        "g_relative_frobenius_K": float(np.mean(grouped["K"])),
        "g_relative_frobenius_k_generic_1": float(np.mean(grouped["k_generic_1"])),
    }


def scatter_plot(plot_id: str, title: str, y_title: str, metric: str,
                 rows: list[dict], metrics: list[dict] | None = None) -> dict:
    return {
        "id": plot_id, "kind": "scatter", "title": title,
        "subtitle": "Graphene Γ E2g electron-phonon derivative diagnostics",
        "reference_label": "Reference: frozen SIESTA dH/dR",
        "x_title": "N_train snapshots", "y_title": y_title,
        "x_key": "x_dataset_size", "y_key": metric,
        "series_key": "model_label", "rows": rows,
        **({"metrics": metrics} if metrics else {}),
    }


def write_ui_payload(rows: list[dict]) -> None:
    metric_rows = [{
        **row,
        "model": row.get("model", "graph2mat"),
        "model_label": (
            f"DeepH { {'w90': 'W90', '5x5': '5×5'}[row['family']] }"
            if row.get("model") == "deeph"
            else {"w90": "W90", "5x5": "5×5", "w90_adjusted": "W90 ajustado",
                  "5x5_adjusted": "5×5 ajustado"}[row["family"]]
        ),
        "x_dataset_size": row["snapshots"], "x_dataset_size_kind": "N_train",
        "n_train": row["snapshots"], "n_total": row["snapshots"],
        "dataset_size_source": "checkpoint_training_campaign",
        "dataset_ids": [f"graphene_{row['family']}_{row['snapshots']}"],
        "n_rows": len(DIRECTIONS), "n_stencils": len(DIRECTIONS),
        "delta_values": [0.02], "axes": list(DIRECTIONS),
    } for row in rows]
    specs = [
        ("dh_mae_vs_dataset_size", "dH MAE vs dataset size", "dH MAE eV/Ang", "dh_mae_union_eV_per_Ang", None),
        ("dh_rmse_vs_dataset_size", "dH RMSE vs dataset size", "dH RMSE eV/Ang", "dh_rmse_union_eV_per_Ang", None),
        ("relative_frobenius_vs_dataset_size", "Relative Frobenius vs dataset size", "Relative Frobenius", "dh_relative_frobenius_ref", None),
        ("support_f1_vs_dataset_size", "Support F1 vs dataset size", "Support F1", "dh_support_f1", None),
        ("support_error_rates_vs_dataset_size", "Support error rates vs dataset size", "Support error rate", "dh_false_zero_rate", [
            {"key": "dh_false_zero_rate", "label": "False-zero rate", "unit": ""},
            {"key": "dh_false_nonzero_rate", "label": "False-nonzero rate", "unit": ""}]),
        ("robust_relative_frobenius_vs_dataset_size", "Robust relative Frobenius vs dataset size", "Robust relative Frobenius", "dh_relative_frobenius_union_robust", None),
        ("robust_relative_l1_vs_dataset_size", "Robust relative L1 vs dataset size", "Robust relative L1", "dh_relative_l1_union_robust", None),
        ("derivative_correlation_vs_dataset_size", "Derivative correlation vs dataset size", "Pearson correlation", "dh_pearson_union", None),
        ("derivative_residual_summary_vs_dataset_size", "Derivative residual summary vs dataset size", "Residual eV/Ang", "dh_residual_mean_union_eV_per_Ang", [
            {"key": "dh_residual_mean_union_eV_per_Ang", "label": "Mean residual", "unit": "eV/Ang"},
            {"key": "dh_residual_std_union_eV_per_Ang", "label": "Residual std", "unit": "eV/Ang"},
            {"key": "dh_residual_median_union_eV_per_Ang", "label": "Median residual", "unit": "eV/Ang"}]),
        ("derivative_residual_tail_vs_dataset_size", "Derivative residual tail vs dataset size", "Absolute residual eV/Ang", "dh_residual_abs_p90_union_eV_per_Ang", [
            {"key": "dh_residual_abs_p90_union_eV_per_Ang", "label": "p90", "unit": "eV/Ang"},
            {"key": "dh_residual_abs_p95_union_eV_per_Ang", "label": "p95", "unit": "eV/Ang"},
            {"key": "dh_residual_abs_p99_union_eV_per_Ang", "label": "p99", "unit": "eV/Ang"}]),
        ("epc_g_relative_error_K_vs_dataset_size", "EPC g relative error at K", "Relative Frobenius", "g_relative_frobenius_K", None),
        ("epc_g_relative_error_generic_k_vs_dataset_size", "EPC g relative error at generic k", "Relative Frobenius", "g_relative_frobenius_k_generic_1", None),
    ]
    plots = [scatter_plot(plot_id, title, y_title, metric, metric_rows, metrics)
             for plot_id, title, y_title, metric, metrics in specs]
    payload = {
        "schema": "hamiltonian_derivative_plot_payload_v1", "available": True,
        "diagnostic_only": True, "scientific_status": "diagnostic_only",
        "title": "Graphene Γ E2g electron-phonon derivative diagnostics",
        "reference_label": "Reference: frozen SIESTA dH/dR",
        "force_constants_label": "E2g electronic derivatives; these are not force constants",
        "units": {"derivative": "eV/Ang", "delta": "Ang"},
        "methods_seen": sorted({row.get("model", "graph2mat") for row in rows}),
        "method_label": "Graph2Mat + DeepH" if any(row.get("model") == "deeph" for row in rows) else "Graph2Mat",
        "plots": plots, "primary_plot_ids": [spec[0] for spec in specs],
        "dataset_size_plot_ids": [spec[0] for spec in specs], "diagnostic_plot_ids": [],
        "scientific_warnings": [{"severity": "warning", "code": "epc_matrix_proxy",
            "message": "Matrix dH/dR metrics diagnose the E2g electronic derivative; they are not a direct error bar on the final EPC g."}],
        "summary": {"metric_rows": len(metric_rows), "dataset_size_rows": len(metric_rows)},
    }
    graph_root = UI_OUTPUT / "derivative_metrics/graph2mat"
    summary_root = UI_OUTPUT / "derivative_metrics/summary/derivative_plots"
    graph_root.mkdir(parents=True, exist_ok=True); summary_root.mkdir(parents=True, exist_ok=True)
    (graph_root / "manifest.json").write_text(json.dumps({
        "scientific_status": "diagnostic_only", "finite_difference_method": "graph2mat_jvp",
        "derivative_units": "eV/Ang", "stencils_ok": len(metric_rows), "stencils_failed": 0,
        "warnings": payload["scientific_warnings"], "fatal_errors": []}, indent=2) + "\n")
    with (graph_root / "derivative_matrix_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys(), extrasaction="ignore")
        writer.writeheader(); writer.writerows(metric_rows)
    (summary_root / "derivative_plot_payload.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in ("w90", "5x5"):
        for size in SIZES:
            source = checkpoint(family, size)
            run = evaluate(family, size, source)
            metrics = matrix_metrics(run)
            rows.append({"family": family, "snapshots": size, **metrics, **g_metrics(run),
                         "mae_eV_per_ang": metrics["dh_mae_union_eV_per_Ang"],
                         "relative_frobenius": metrics["dh_relative_frobenius_ref"],
                         "checkpoint": str(source)})
            print(f"{family}-{size}: MAE={metrics['dh_mae_union_eV_per_Ang']:.6g}, relative Frobenius={metrics['dh_relative_frobenius_ref']:.6g}", flush=True)

    for family, snapshots, root in (
        ("w90_adjusted", 500, ROOT / "Comparison/results/epc/derivative_remediation_w90_500_20260908"),
        ("5x5_adjusted", 400, ROOT / "Comparison/results/epc/derivative_remediation_5x5_400_coordinates_blended_20260909"),
    ):
        metrics = matrix_metrics(root / "final_epc_evaluation")
        rows.append({"family": family, "snapshots": snapshots, **metrics, **g_metrics(root / "final_epc_evaluation"),
                     "mae_eV_per_ang": metrics["dh_mae_union_eV_per_Ang"],
                     "relative_frobenius": metrics["dh_relative_frobenius_ref"],
                     "checkpoint": str(root / "selected_derivative_remediated.ckpt")})

    with (OUTPUT / "epc_snapshot_scaling_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (OUTPUT / "epc_snapshot_scaling_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    write_ui_payload(rows)

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for family, label in (("w90", "W90"), ("5x5", "5×5")):
        selected = [row for row in rows if row["family"] == family]
        axes[0].plot([row["snapshots"] for row in selected],
                     [row["mae_eV_per_ang"] for row in selected], "o-", label=label)
        axes[1].plot([row["snapshots"] for row in selected],
                     [100 * row["relative_frobenius"] for row in selected], "o-", label=label)
    for star, label, color in ((rows[-2], "W90-500 ajustado", "black"),
                               (rows[-1], "5×5-400 ajustado", "#7c3aed")):
        axes[0].scatter(star["snapshots"], star["mae_eV_per_ang"], marker="*", s=150,
                        color=color, label=label, zorder=5)
        axes[1].scatter(star["snapshots"], 100 * star["relative_frobenius"], marker="*", s=150,
                        color=color, label=label, zorder=5)
    axes[0].set(xlabel="Snapshots de entrenamiento", ylabel="MAE de dH/dR (eV/Å)",
                title="Error absoluto E2g")
    axes[1].set(xlabel="Snapshots de entrenamiento", ylabel="Frobenius relativo (%)",
                title="Error relativo E2g")
    for axis in axes:
        axis.grid(alpha=0.25); axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "epc_precision_vs_snapshots.png", dpi=200)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
