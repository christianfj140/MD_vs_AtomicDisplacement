#!/usr/bin/env python3
"""Plot final Graph2Mat/DeepH valence-band diagnostics and Graph2Mat RMSE matrices.

The band plots use already materialized final-test eigenvalue CSV files. Energies
are plotted relative to the SIESTA Fermi level using the stored band-error CSVs:

    E_pred - E_F = (E_SIESTA - E_F) + (E_pred - E_SIESTA)

This avoids relying on adapter-specific absolute-energy offsets. The k-axis is
the repository k-point order from the materialized Monkhorst-Pack grid, not a
high-symmetry path unless such a path was explicitly used upstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hamiltonian_metrics import read_matrix  # noqa: E402


DEFAULT_DATASETS = {
    "iid600": {
        "workflow_root": Path("Comparison/results/g2m_deeph_iid600_phaseB_intermediate_spectral_refine_v1"),
        "run_name": "paper_ready_final70_iid600_20260602_123539",
        "dataset_id": "graphene_w90_phase1_iid600",
    },
    "iid1000": {
        "workflow_root": Path("Comparison/results/g2m_deeph_iid1000_phaseB_transfer_spectral_refine_v1"),
        "run_name": "paper_ready_final70_iid1000_20260602_123539",
        "dataset_id": "graphene_w90_phase1_iid1000",
    },
}


@dataclass(frozen=True)
class WinnerRun:
    dataset_key: str
    dataset_id: str
    model: str
    selected_config_id: str
    config_id: str
    run_dir: Path
    metric_value: float | None
    seed: int | None
    eigen_dir: Path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def final_stats_winner_config(workflow_root: Path, model: str) -> str:
    stats_path = workflow_root / "final_test" / "final_statistics.json"
    stats = read_json(stats_path)
    decisions = stats.get("winner_decision", {}).get("dataset_decisions") or []
    if not decisions:
        raise RuntimeError(f"No dataset_decisions in {stats_path}")
    best_by_model = decisions[0].get("best_config_by_model") or {}
    record = best_by_model.get(model)
    if not isinstance(record, dict) or not record.get("selected_config_id"):
        raise RuntimeError(f"No best {model} config in {stats_path}")
    return str(record["selected_config_id"])


def run_metric_value(row: dict[str, Any]) -> float | None:
    for key in ("final_test_metric_value", "low_energy_rmse_eV"):
        value = finite_float(row.get(key))
        if value is not None:
            return value
    metrics = row.get("final_test_metrics") or {}
    if isinstance(metrics, dict):
        return finite_float(metrics.get("low_energy_rmse_eV"))
    return None


def choose_seed_run(workflow_root: Path, run_name: str, dataset_key: str, dataset_id: str, model: str) -> WinnerRun:
    selected_config_id = final_stats_winner_config(workflow_root, model)
    manifest_path = workflow_root / "final_test" / "sweep" / "training_sweep_manifest.json"
    manifest = read_json(manifest_path)
    candidates = [
        row
        for row in manifest.get("runs", [])
        if row.get("model") == model
        and row.get("dataset_id") == dataset_id
        and row.get("selected_config_id") == selected_config_id
        and str(row.get("status", "")).lower() == "completed"
    ]
    if not candidates:
        raise RuntimeError(f"No completed {model} runs for {dataset_id} {selected_config_id} in {manifest_path}")
    candidates.sort(key=lambda row: (run_metric_value(row) is None, run_metric_value(row) or float("inf")))
    row = candidates[0]
    config_id = str(row.get("config_id") or "")
    if not config_id:
        raise RuntimeError(f"Selected run for {model} lacks config_id in {manifest_path}")
    run_dir = workflow_root / "runs" / run_name / "sweep" / model / dataset_id / config_id
    if model == "graph2mat":
        eigen_dir = run_dir / "metrics" / "graph2mat" / "test_eval_input" / "eigenvalues"
    elif model == "deeph":
        eigen_dir = run_dir / "metrics" / "deeph" / "test_eval" / "eigenvalues"
    else:  # pragma: no cover
        raise ValueError(model)
    if not eigen_dir.exists():
        raise RuntimeError(f"Missing eigenvalue directory for {model}: {eigen_dir}")
    seed = None
    common = row.get("common")
    if isinstance(common, dict):
        try:
            seed = int(common.get("seed"))
        except (TypeError, ValueError):
            seed = None
    return WinnerRun(
        dataset_key=dataset_key,
        dataset_id=dataset_id,
        model=model,
        selected_config_id=selected_config_id,
        config_id=config_id,
        run_dir=run_dir,
        metric_value=run_metric_value(row),
        seed=seed,
        eigen_dir=eigen_dir,
    )


def sample_order(eigen_dir: Path) -> list[str]:
    kpoints_path = eigen_dir / "kpoints.csv"
    seen: dict[str, None] = {}
    with kpoints_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample = str(row.get("sample") or "").strip()
            if sample:
                seen.setdefault(sample, None)
    return list(seen)


def common_sample(winners: list[WinnerRun], requested: str | None) -> str:
    sample_sets = []
    for winner in winners:
        samples = sample_order(winner.eigen_dir)
        if not samples:
            raise RuntimeError(f"No samples in {winner.eigen_dir / 'kpoints.csv'}")
        sample_sets.append(samples)
    if requested:
        if all(requested in samples for samples in sample_sets):
            return requested
        raise RuntimeError(f"Requested sample {requested!r} is not materialized for every winner run")
    common = set(sample_sets[0])
    for samples in sample_sets[1:]:
        common &= set(samples)
    if not common:
        raise RuntimeError("No common materialized sample across winner runs")
    for sample in sample_sets[0]:
        if sample in common:
            return sample
    raise RuntimeError("No common sample found")


def kpoint_indices_for_sample(eigen_dir: Path, sample: str) -> list[int]:
    indices: list[int] = []
    with (eigen_dir / "kpoints.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("sample")) != sample:
                continue
            value = row.get("k_index")
            if value is None:
                value = row.get("k")
            indices.append(int(float(value)))
    return sorted(indices)


def read_band_error_file(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            band = int(float(row["band"]))
            siesta_minus_fermi = float(row["siesta_minus_fermi_eV"])
            error = float(row["error_eV"])
            rows.append(
                {
                    "band": float(band),
                    "siesta_minus_fermi_eV": siesta_minus_fermi,
                    "pred_minus_fermi_eV": siesta_minus_fermi + error,
                    "error_eV": error,
                }
            )
    rows.sort(key=lambda item: int(item["band"]))
    return rows


def load_sample_bands(eigen_dir: Path, sample: str, max_kpoints: int | None) -> dict[int, dict[str, list[float]]]:
    result: dict[int, dict[str, list[float]]] = {}
    indices = kpoint_indices_for_sample(eigen_dir, sample)
    if max_kpoints is not None:
        indices = indices[:max_kpoints]
    for k_index in indices:
        path = eigen_dir / "kpoint_band_errors" / f"{sample}_k{k_index:04d}.csv"
        if not path.exists():
            continue
        for row in read_band_error_file(path):
            band = int(row["band"])
            store = result.setdefault(
                band,
                {"k_index": [], "siesta_minus_fermi_eV": [], "pred_minus_fermi_eV": [], "error_eV": []},
            )
            store["k_index"].append(k_index)
            store["siesta_minus_fermi_eV"].append(row["siesta_minus_fermi_eV"])
            store["pred_minus_fermi_eV"].append(row["pred_minus_fermi_eV"])
            store["error_eV"].append(row["error_eV"])
    return result


def valence_band_ids(reference_bands: dict[int, dict[str, list[float]]], max_valence_bands: int) -> list[int]:
    valence: list[tuple[float, int]] = []
    for band, values in reference_bands.items():
        arr = np.asarray(values["siesta_minus_fermi_eV"], dtype=float)
        if arr.size and float(np.nanmedian(arr)) <= 0.0:
            valence.append((float(np.nanmedian(arr)), band))
    valence.sort(key=lambda item: item[0], reverse=True)
    chosen = [band for _, band in valence[:max_valence_bands]]
    return sorted(chosen)


def plot_valence_bands(
    dataset_key: str,
    sample: str,
    g2m: WinnerRun,
    deeph: WinnerRun,
    output_dir: Path,
    *,
    max_valence_bands: int,
    max_kpoints: int | None,
) -> dict[str, Any]:
    g2m_bands = load_sample_bands(g2m.eigen_dir, sample, max_kpoints)
    deeph_bands = load_sample_bands(deeph.eigen_dir, sample, max_kpoints)
    band_ids = valence_band_ids(g2m_bands, max_valence_bands)
    if not band_ids:
        raise RuntimeError(f"No valence bands found for {dataset_key} sample {sample}")

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    colors = {"graph2mat": "#1f77b4", "deeph": "#d62728"}
    labels_done = {"siesta": False, "graph2mat": False, "deeph": False}
    for band in band_ids:
        ref = g2m_bands.get(band)
        if not ref:
            continue
        ref_k = np.asarray(ref["k_index"], dtype=int)
        ref_energy = np.asarray(ref["siesta_minus_fermi_eV"], dtype=float)
        ax.plot(
            ref_k,
            ref_energy,
            color="black",
            linewidth=1.7,
            alpha=0.9,
            label="SIESTA" if not labels_done["siesta"] else None,
        )
        labels_done["siesta"] = True
        pred = g2m_bands.get(band)
        if pred:
            pred_energy = ref_energy + np.asarray(pred["error_eV"], dtype=float)
            ax.plot(
                ref_k,
                pred_energy,
                color=colors["graph2mat"],
                linewidth=1.35,
                linestyle="--",
                alpha=0.88,
                label=f"Graph2Mat {g2m.selected_config_id}" if not labels_done["graph2mat"] else None,
            )
            labels_done["graph2mat"] = True
        pred = deeph_bands.get(band)
        if pred:
            pred_k = np.asarray(pred["k_index"], dtype=int)
            pred_error = np.asarray(pred["error_eV"], dtype=float)
            if pred_k.shape != ref_k.shape or not np.array_equal(pred_k, ref_k):
                common_errors = dict(zip(pred_k.tolist(), pred_error.tolist()))
                pred_error = np.asarray([common_errors.get(int(k), np.nan) for k in ref_k], dtype=float)
            pred_energy = ref_energy + pred_error
            ax.plot(
                ref_k,
                pred_energy,
                color=colors["deeph"],
                linewidth=1.35,
                linestyle=(0, (5, 2)),
                alpha=0.88,
                label=f"DeepH {deeph.selected_config_id}" if not labels_done["deeph"] else None,
            )
            labels_done["deeph"] = True
    ax.axhline(0.0, color="#555555", linewidth=0.9, alpha=0.55)
    ax.set_title(f"{dataset_key}: valence bands vs SIESTA · {sample}")
    ax.set_xlabel("k-point index from materialized Monkhorst-Pack grid")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.grid(True, color="#e6e9ef", linewidth=0.8)
    ax.legend(frameon=False, ncols=3, fontsize=9)
    ax.text(
        0.01,
        0.015,
        "Diagnostic k-grid order, not a high-symmetry path unless the upstream k-points define one.",
        transform=ax.transAxes,
        fontsize=8,
        color="#59636e",
    )
    fig.tight_layout()
    png = output_dir / f"{dataset_key}_valence_bands_{sample}.png"
    pdf = output_dir / f"{dataset_key}_valence_bands_{sample}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "dataset": dataset_key,
        "sample": sample,
        "bands": band_ids,
        "png": str(png),
        "pdf": str(pdf),
        "note": (
            "Energies are relative to the common SIESTA Fermi level from the Graph2Mat/SIESTA "
            "test reference; predictor lines are common SIESTA + model error. The x-axis is "
            "materialized k-grid order."
        ),
    }


def graph2mat_matrix_paths(run: WinnerRun, sample: str) -> tuple[Path, Path]:
    root = run.run_dir / "metrics" / "graph2mat" / "test_eval_input"
    pred = root / "predicted_hamiltonians" / sample / "ML_prediction.HSX"
    ref = root / "siesta_hamiltonians" / sample / "graphene.TSHS"
    if not pred.exists():
        raise RuntimeError(f"Missing Graph2Mat prediction: {pred}")
    if not ref.exists():
        raise RuntimeError(f"Missing SIESTA reference: {ref}")
    return ref, pred


def matrix_rmse_for_samples(run: WinnerRun, samples: list[str]) -> tuple[np.ndarray, list[str]]:
    accum: np.ndarray | None = None
    count = 0
    used: list[str] = []
    for sample in samples:
        try:
            ref_path, pred_path = graph2mat_matrix_paths(run, sample)
            ref = read_matrix(ref_path).hamiltonian.toarray()
            pred = read_matrix(pred_path).hamiltonian.toarray()
        except Exception:
            continue
        delta = np.asarray(pred - ref)
        sq = np.abs(delta) ** 2
        if accum is None:
            accum = np.zeros_like(sq, dtype=float)
        if accum.shape != sq.shape:
            continue
        accum += sq
        count += 1
        used.append(sample)
    if accum is None or count == 0:
        raise RuntimeError(f"No matrix RMSE samples could be read for {run.config_id}")
    return np.sqrt(accum / count), used


def write_matrix_csv(path: Path, matrix_mev: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_col", *[f"col_{i}" for i in range(matrix_mev.shape[1])]])
        for row_idx, row in enumerate(matrix_mev):
            writer.writerow([f"row_{row_idx}", *[f"{float(value):.10g}" for value in row]])


def plot_graph2mat_rmse_matrix(
    dataset_key: str,
    run: WinnerRun,
    output_dir: Path,
    *,
    sample_limit: int,
) -> dict[str, Any]:
    samples = sample_order(run.eigen_dir)[:sample_limit]
    rmse_eV, used_samples = matrix_rmse_for_samples(run, samples)
    rmse_meV = rmse_eV * 1000.0
    n_rows, n_cols = rmse_meV.shape
    fig_width = min(18.0, max(10.0, 0.055 * n_cols + 2.5))
    fig_height = min(7.0, max(3.8, 0.42 * n_rows + 1.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(rmse_meV, cmap="magma", origin="upper", aspect="auto")
    ax.set_title(f"{dataset_key}: Graph2Mat raw-matrix RMSE · {run.selected_config_id}")
    ax.set_xlabel("Column raw sparse index")
    ax.set_ylabel("Row orbital index")
    x_step = max(1, int(math.ceil(n_cols / 12)))
    ax.set_xticks(np.arange(0, n_cols, x_step))
    ax.set_yticks(np.arange(n_rows))
    ax.tick_params(labelsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE meV")
    fig.text(
        0.12,
        0.02,
        f"Aggregated over {len(used_samples)} test samples; raw SIESTA/Graph2Mat matrix shape {rmse_meV.shape}.",
        fontsize=8,
        color="#59636e",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    stem = f"{dataset_key}_graph2mat_rmse_matrix_{run.selected_config_id}_{run.config_id}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    csv_path = output_dir / f"{stem}.csv"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    write_matrix_csv(csv_path, rmse_meV)
    return {
        "dataset": dataset_key,
        "selected_config_id": run.selected_config_id,
        "config_id": run.config_id,
        "samples_used": used_samples,
        "rmse_matrix_shape": list(rmse_meV.shape),
        "rmse_meV_mean": float(np.mean(rmse_meV)),
        "rmse_meV_max": float(np.max(rmse_meV)),
        "png": str(png),
        "pdf": str(pdf),
        "csv": str(csv_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Comparison/results/winner_visualizations") / datetime.now().strftime("bands_rmse_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--sample-id", default=None, help="Optional common sample id, e.g. md_540.")
    parser.add_argument("--max-kpoints", type=int, default=None)
    parser.add_argument("--max-valence-bands", type=int, default=3)
    parser.add_argument("--matrix-sample-limit", type=int, default=20)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_DATASETS), action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    dataset_keys = args.dataset or list(DEFAULT_DATASETS)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "datasets": {},
        "warnings": [],
    }
    for dataset_key in dataset_keys:
        spec = DEFAULT_DATASETS[dataset_key]
        workflow_root = spec["workflow_root"]
        dataset_id = spec["dataset_id"]
        run_name = spec["run_name"]
        g2m = choose_seed_run(workflow_root, run_name, dataset_key, dataset_id, "graph2mat")
        deeph = choose_seed_run(workflow_root, run_name, dataset_key, dataset_id, "deeph")
        sample = common_sample([g2m, deeph], args.sample_id)
        dataset_record: dict[str, Any] = {
            "dataset_id": dataset_id,
            "graph2mat": {
                "selected_config_id": g2m.selected_config_id,
                "config_id": g2m.config_id,
                "seed": g2m.seed,
                "low_energy_rmse_eV": g2m.metric_value,
                "run_dir": str(g2m.run_dir),
            },
            "deeph": {
                "selected_config_id": deeph.selected_config_id,
                "config_id": deeph.config_id,
                "seed": deeph.seed,
                "low_energy_rmse_eV": deeph.metric_value,
                "run_dir": str(deeph.run_dir),
            },
            "sample": sample,
        }
        dataset_record["valence_bands"] = plot_valence_bands(
            dataset_key,
            sample,
            g2m,
            deeph,
            args.output_dir,
            max_valence_bands=args.max_valence_bands,
            max_kpoints=args.max_kpoints,
        )
        try:
            dataset_record["graph2mat_rmse_matrix"] = plot_graph2mat_rmse_matrix(
                dataset_key,
                g2m,
                args.output_dir,
                sample_limit=args.matrix_sample_limit,
            )
        except Exception as exc:
            warning = f"{dataset_key}: Graph2Mat RMSE matrix unavailable: {exc}"
            manifest["warnings"].append(warning)
            dataset_record["graph2mat_rmse_matrix"] = {"available": False, "reason": str(exc)}
        manifest["datasets"][dataset_key] = dataset_record
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
