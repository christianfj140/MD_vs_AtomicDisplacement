"""Aggregate mixing-sweep records into MAE vs total-dataset-size curves.

One curve per ``(mode, ratio, model)``. Emits a JSON payload (for the UI chart)
and, optionally, a PNG (matplotlib import guarded so the JSON path needs no
plotting backend).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .mixing_sweep import _ratio_slug


# Repo-wide statistical bar for claims (mirrors g2m_deeph_final_stats
# --min-final-seeds): curves built from fewer seeds are exploratory only.
MIN_SEEDS_FOR_CLAIMS = 3


def aggregate_mae_vs_size(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Group ``h_mae_eV`` records into MAE-vs-total-size curves.

    Each record needs ``mode``, ``ratio``, ``model``, ``total_size``,
    ``h_mae_eV`` and ideally ``seed``. Points within a curve are sorted by
    total size; replicate values per point (one per seed) are aggregated as
    mean ± sample std with the replicate count ``n_seeds``. Points/curves
    backed by fewer than ``MIN_SEEDS_FOR_CLAIMS`` seeds are flagged
    ``exploratory`` — treat them as exploratory, not publishable evidence.
    """
    curves: dict[tuple[str, float, str], dict[str, dict[str, Any]]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            key = (
                str(record["mode"]),
                round(float(record["ratio"]), 6),
                str(record["model"]),
            )
            size = int(record.get("size", record["total_size"]))
            mode = str(record["mode"])
            ratio = round(float(record["ratio"]), 6)
            total_size = int(record["total_size"])
            mae = float(record["h_mae_eV"])
        except (KeyError, TypeError, ValueError):
            continue
        # Derive from (size, mode, ratio), not output_root: output_root is a
        # deterministic function of this triple but its full path varies
        # across machines/tmp dirs, which would break id-based merging with
        # payload lists computed elsewhere (see pipeline_ui._mixing_payload_id).
        payload_id = str(record.get("payload_id") or f"size{size}_{mode}_{_ratio_slug(ratio)}")
        payloads.setdefault(
            payload_id,
            {
                "id": payload_id,
                "label": record.get("payload_label") or f"size={size} {mode} ratio={ratio:g}",
                "size": size,
                "mode": mode,
                "ratio": ratio,
                "total_size": total_size,
                "output_root": record.get("output_root"),
            },
        )
        bucket = curves.setdefault(key, {}).setdefault(
            payload_id,
            {
                "payload_id": payload_id,
                "total_size": total_size,
                "values": [],
                "seeds": set(),
            },
        )
        bucket["values"].append(mae)
        if record.get("seed") is not None:
            try:
                bucket["seeds"].add(int(record["seed"]))
            except (TypeError, ValueError):
                pass

    def _point(item: dict[str, Any]) -> dict[str, Any]:
        values = item["values"]
        mean = sum(values) / len(values)
        if len(values) > 1:
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        else:
            std = None
        # Replicates = distinct seeds when recorded, else raw value count.
        n_seeds = len(item["seeds"]) if item["seeds"] else len(values)
        return {
            "payload_id": item["payload_id"],
            "total_size": item["total_size"],
            "mae": mean,
            "mae_std": std,
            "n_seeds": n_seeds,
            "exploratory": n_seeds < MIN_SEEDS_FOR_CLAIMS,
        }

    curve_list: list[dict[str, Any]] = []
    for (mode, ratio, model), by_payload in sorted(curves.items()):
        points = [
            _point(item)
            for item in sorted(by_payload.values(), key=lambda value: (value["total_size"], value["payload_id"]))
        ]
        curve_exploratory = any(point["exploratory"] for point in points)
        curve_list.append(
            {
                "mode": mode,
                "ratio": ratio,
                "model": model,
                "label": f"{model} · {mode} · ratio={ratio:g}",
                "exploratory": curve_exploratory,
                "points": points,
            }
        )
    exploratory = any(curve["exploratory"] for curve in curve_list)
    warnings: list[str] = []
    if exploratory and curve_list:
        warnings.append(
            f"curves with fewer than {MIN_SEEDS_FOR_CLAIMS} seeds per point are "
            "EXPLORATORY: seed-to-seed training variance is not resolved, do "
            "not use them for publishable composition claims"
        )
    return {
        "schema": "ml_vs_siesta_mae_vs_size_v2",
        "metric": "h_mae_eV",
        "x": "total_dataset_size",
        "min_seeds_for_claims": MIN_SEEDS_FOR_CLAIMS,
        "exploratory": exploratory,
        "warnings": warnings,
        "n_curves": len(curve_list),
        "payloads": sorted(payloads.values(), key=lambda item: (item["size"], item["mode"], item["ratio"], item["id"])),
        "curves": curve_list,
    }


def build_mae_vs_size_from_sweep(sweep_summary: dict[str, Any]) -> dict[str, Any]:
    """Aggregate directly from a ``run_mixing_sweep`` summary dict."""
    return aggregate_mae_vs_size(sweep_summary.get("records") or [])


def plot_mae_vs_size(aggregated: dict[str, Any], output_png: str | Path) -> str:
    """Render the aggregated curves to a PNG. Returns the written path."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("matplotlib is required to plot MAE vs size.") from exc

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for curve in aggregated.get("curves", []):
        points = curve.get("points") or []
        if not points:
            continue
        xs = [p["total_size"] for p in points]
        ys = [p["mae"] for p in points]
        yerr = [p.get("mae_std") or 0.0 for p in points]
        linestyle = "--" if curve.get("mode") == "replace" else "-"
        ax.errorbar(
            xs, ys, yerr=yerr if any(yerr) else None,
            marker="o", linestyle=linestyle, capsize=3, label=curve.get("label"),
        )
    ax.set_xlabel("Total dataset size (snapshots)")
    ax.set_ylabel("Hamiltonian MAE (eV)")
    title = "MAE vs dataset size — small/large mixing"
    if aggregated.get("exploratory"):
        title += f"  [EXPLORATORY: <{aggregated.get('min_seeds_for_claims', 3)} seeds/point]"
    ax.set_title(title)
    if aggregated.get("curves"):
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return str(output_png)


def write_mae_vs_size_outputs(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    write_png: bool = True,
) -> dict[str, Any]:
    """Write ``mae_vs_size.json`` (+ optional PNG) and return the aggregate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate_mae_vs_size(records)
    json_path = output_dir / "mae_vs_size.json"
    json_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    aggregated["json_path"] = str(json_path)
    if write_png and aggregated["curves"]:
        try:
            aggregated["png_path"] = plot_mae_vs_size(aggregated, output_dir / "mae_vs_size.png")
        except RuntimeError:
            aggregated["png_path"] = None
    return aggregated
