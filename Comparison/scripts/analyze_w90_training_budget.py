#!/usr/bin/env python3
"""Compare 2k/4k w90 checkpoints on the existing test, paired by seed and structure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataset_design_curves_v1 as c


def main():
    samples = c.read_json(c.OUT / "final_test_manifest.json")["systems"]["w90"]["samples"]
    ids = [s["sample_id"] for s in samples]
    stages = ("no_early_stop", "no_early_stop_4k")
    metrics = ("H_MAE_meV", "rel_Frob", "band_rmse_meV", "dos_rel_L1")
    masks = {"all": np.ones(len(ids), dtype=bool)}
    masks.update({origin: np.array([s["origin"] == origin for s in samples])
                  for origin in ("synthetic", "md")})
    values = {}
    seed_rows = []
    for stage in stages:
        models = sorted((r for r in c.load_results("w90") if r["stage"] == stage),
                        key=lambda r: r["training_seed"])
        assert [r["training_seed"] for r in models] == list(range(5))
        per_seed = []
        for r in models:
            path = c.OUT / "runs/w90" / r["job_id"] / "final_test/per_structure_metrics.csv"
            rows = c.read_csv(path)
            by_id = {row["sample_id"]: row for row in rows}
            assert len(rows) == len(by_id) == len(ids) and set(by_id) == set(ids)
            data = np.array([[float(by_id[sid][m]) for m in metrics] for sid in ids])
            assert np.isfinite(data).all()
            per_seed.append(data)
            for domain, mask in masks.items():
                seed_rows.append({"stage": stage, "seed": r["training_seed"], "domain": domain,
                                  "n_structures": int(mask.sum()), "train_minutes": r["train_seconds"] / 60,
                                  **dict(zip(metrics, data[mask].mean(axis=0)))})
        values[stage] = np.stack(per_seed)
    summary = []
    for domain, mask in masks.items():
        a, b = [values[stage][:, mask].mean(axis=1) for stage in stages]
        for i, metric in enumerate(metrics):
            delta = b[:, i] - a[:, i]
            summary.append({"domain": domain, "metric": metric, "n_seeds": 5,
                            "n_structures": int(mask.sum()),
                            "mean_2k": a[:, i].mean(), "sd_2k": a[:, i].std(ddof=1),
                            "mean_4k": b[:, i].mean(), "sd_4k": b[:, i].std(ddof=1),
                            "mean_paired_delta": delta.mean(),
                            "improved_seeds": int((delta < 0).sum()),
                            "reduction_pct": 100 * (1 - b[:, i].mean() / a[:, i].mean())})
    out = c.OUT / "w90_budget_physics"
    out.mkdir(exist_ok=True)
    c.write_csv(out / "per_seed.csv", seed_rows)
    c.write_csv(out / "summary.csv", summary)
    amplitude_rows = []
    amplitudes = np.array([s["real_amplitude_ang"] for s in samples])
    for stage in stages:
        for amplitude in c.FINAL_AMPLITUDES:
            mask = masks["synthetic"] & np.isclose(amplitudes, amplitude, atol=1e-6)
            data = values[stage][:, mask].mean(axis=1)
            amplitude_rows.append({"stage": stage, "amplitude_ang": amplitude,
                                   "n_structures": int(mask.sum()),
                                   **{m: data[:, i].mean() for i, m in enumerate(metrics)}})
    c.write_csv(out / "by_amplitude.csv", amplitude_rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for ax, i, label, scale in zip(axes, (0, 2, 3), ("H-MAE (meV)", "Band RMSE (meV)", "DOS relative L1 (%)"), (1, 1, 100)):
        for seed in range(5):
            ax.plot([2000, 4000], [values[s][seed, :, i].mean() * scale for s in stages],
                    "o-", linewidth=1, markersize=4, label=f"Seed {seed}")
        ax.set(xticks=[2000, 4000], xlabel="Training epochs", ylabel=label)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("w90: existing common test (48 synthetic + 16 MD structures)")
    fig.savefig(out / "paired_physics.png", dpi=180)
    fig.savefig(out / "paired_physics.pdf")
    plt.close(fig)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
