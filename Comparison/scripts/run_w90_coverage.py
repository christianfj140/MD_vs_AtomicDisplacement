#!/usr/bin/env python3
"""Scientific follow-up: label the expanded pool, train, then evaluate all arms."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataset_design_curves_v1 as c


def prepare_pool():
    path = c.OUT / "coverage_pool_w90_r012.json"
    parent, rows = c._finalist("w90", "efficient")
    assert parent["dim"] == "3D" and parent["R"] == 0.08 and len(rows) == 64
    geometry = c.sampler.load_graphene_primitive()
    base = c.base_positions("w90")
    assert np.allclose(geometry.positions_ang, base, atol=1e-8)
    active = c.sampler.select_active_atoms(geometry, 2)
    forbidden = {c.geometry_record("w90", s.run_fdf)["hash"] for s in c.dev_samples("w90")}
    forbidden |= {s["hash"] for s in c.read_json(c.OUT / "final_test_manifest.json")["systems"]["w90"]["samples"]}
    entries, expected = [], []
    for i, row in enumerate(rows):
        positions = np.asarray(c.s5._positions_from_run_fdf(Path(row["run_fdf"])))
        vectors = 1.5 * (positions - base)
        displaced = base + vectors
        expected.append(displaced)
        assert c.s5.geometry_hash(displaced) not in forbidden
        displacements = {j: tuple(v) for j, v in enumerate(vectors)}
        metadata = c.sampler.build_metadata(geometry, active, displacements,
                                            family="random_cartesian", dim="3D", amplitude_ang=0.12)
        entries.append((f"coverage_w90_3D_R012__pt{i:04d}",
                        c.sampler.Configuration("random_cartesian", displacements, metadata)))
    if not path.exists():
        refs = c.label_structures(entries, c.OUT / "coverage_w90_r012", c.sampler.GRAPHENE_PRIMITIVE_FDF, 4)
        samples = [c.labelled_sample("w90", sid, refs, {"family": "random_cartesian", "dim": "3D",
                   "k": 2, "amplitude_ang": 0.12}) for sid, _ in entries]
        hashes = [r["hash"] for r in samples]
        assert len(set(hashes)) == 64 and not set(hashes) & forbidden
        pool = {"recipe_id": "random_cartesian__3D__k2__r0.120__res64__scaled_seed0",
                "family": "random_cartesian", "dim": "3D", "k": 2, "R": 0.12,
                "sampler_seed": parent["sampler_seed"], "pool_hash": c.ids_hash(hashes),
                "parent_recipe_id": parent["recipe_id"], "parent_pool_hash": parent["pool_hash"],
                "displacement_scale": 1.5, "samples": samples}
        c.write_json(path, pool)
    pool = c.read_json(path)
    assert pool["parent_pool_hash"] == parent["pool_hash"]
    assert len(pool["samples"]) == 64
    for row, target in zip(pool["samples"], expected):
        assert np.allclose(c.s5._positions_from_run_fdf(Path(row["run_fdf"])), target, atol=1e-8)
        assert Path(row["reference_matrix"]).exists()
        assert (Path(row["reference_dir"]) / "0_NORMAL_EXIT").exists()
        assert row["hash"] not in forbidden and row["siesta_cpu_s"] > 0
    for job in c.coverage_4k_jobs("w90"):
        assert job["val"] == "dev" and len(job["train"]) == 64
        assert not forbidden & {r["hash"] for r in job["train"]}
    print("Pool verified: 64 labels; paired displacement scale 1.5; train/dev/test disjoint", flush=True)


def summarize():
    samples = c.read_json(c.OUT / "final_test_manifest.json")["systems"]["w90"]["samples"]
    ids = [r["sample_id"] for r in samples]
    metrics = ("H_MAE_meV", "rel_Frob", "band_rmse_meV", "dos_rel_L1")
    models = [r for r in c.load_results("w90") if r["stage"] in ("no_early_stop_4k", "coverage_4k")]
    assert len(models) == 20
    output, cubes = [], {}
    for model in models:
        path = c.OUT / "runs/w90" / model["job_id"] / "final_test/per_structure_metrics.csv"
        rows = c.read_csv(path)
        indexed = {r["sample_id"]: r for r in rows}
        assert len(rows) == len(indexed) == 64 and set(indexed) == set(ids)
        cubes[(model["recipe_id"], model["training_seed"])] = np.array(
            [[float(indexed[sid][m]) for m in metrics] for sid in ids])
        for domain in ("all", "synthetic", "md"):
            selected = [s for s in samples if domain == "all" or s["origin"] == domain]
            values = np.array([[float(indexed[s["sample_id"]][m]) for m in metrics] for s in selected])
            assert np.isfinite(values).all()
            output.append({"recipe": model["recipe_id"], "seed": model["training_seed"],
                           "domain": domain, "N": 64, "n_structures": len(selected),
                           "siesta_cpu_h_train": model["siesta_cpu_h_train"],
                           "train_wall_minutes": model["train_seconds"] / 60,
                           **dict(zip(metrics, values.mean(axis=0)))})
    out = c.OUT / "w90_coverage_physics"
    c.write_csv(out / "per_seed.csv", output)
    summary = []
    for recipe in sorted({r["recipe"] for r in output}):
        for domain in ("all", "synthetic", "md"):
            rows = [r for r in output if r["recipe"] == recipe and r["domain"] == domain]
            assert sorted(r["seed"] for r in rows) == list(range(5))
            row = {"recipe": recipe, "domain": domain, "n_seeds": 5}
            for metric in (*metrics, "siesta_cpu_h_train", "train_wall_minutes"):
                values = [r[metric] for r in rows]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_sd"] = float(np.std(values, ddof=1))
            summary.append(row)
            print(row, flush=True)
    c.write_csv(out / "summary.csv", summary)

    # Paired bootstrap: same training seed and same test structure in both arms.
    recipes = {
        "1D R0.08": "random_cartesian__1D_in__k2__r0.080__res64__seed2",
        "3D R0.08": "random_cartesian__3D__k2__r0.080__res64__seed0",
        "3D R0.12": "random_cartesian__3D__k2__r0.120__res64__scaled_seed0",
        "MD": "md_w90"}
    assert set(recipes.values()) == {r["recipe_id"] for r in models}
    arrays = {label: np.stack([cubes[(recipe, seed)] for seed in range(5)])
              for label, recipe in recipes.items()}
    masks = {"all": np.ones(len(samples), dtype=bool),
             "synthetic": np.array([s["origin"] == "synthetic" for s in samples]),
             "md": np.array([s["origin"] == "md" for s in samples])}
    rng = np.random.default_rng(20260921)
    comparisons = []
    for left, right in (("3D R0.12", "3D R0.08"), ("3D R0.08", "1D R0.08"),
                        ("3D R0.12", "MD"), ("3D R0.08", "MD")):
        for domain, mask in masks.items():
            delta = arrays[left][:, mask] - arrays[right][:, mask]
            boot = np.empty((10_000, len(metrics)))
            for i in range(len(boot)):
                seed_idx = rng.integers(0, 5, 5)
                structure_idx = rng.integers(0, delta.shape[1], delta.shape[1])
                boot[i] = delta[seed_idx][:, structure_idx].mean(axis=(0, 1))
            for j, metric in enumerate(metrics):
                comparisons.append({"A": left, "B": right, "domain": domain, "metric": metric,
                                    "mean_A_minus_B": float(delta[:, :, j].mean()),
                                    "ci95_low": float(np.percentile(boot[:, j], 2.5)),
                                    "ci95_high": float(np.percentile(boot[:, j], 97.5)),
                                    "better_seeds": int((delta[:, :, j].mean(axis=1) < 0).sum())})
    c.write_csv(out / "paired_comparisons.csv", comparisons)

    grouped = []
    synthetic = masks["synthetic"]
    for label, array in arrays.items():
        for field, levels in (("dim", c.DIMS), ("amplitude_ang", c.FINAL_AMPLITUDES)):
            for level in levels:
                mask = synthetic & np.array([s[field] == level or (field == "amplitude_ang" and
                                             np.isclose(s[field], level)) for s in samples])
                values = array[:, mask].mean(axis=1)
                grouped.append({"recipe": label, "group": field, "value": level,
                                "n_structures": int(mask.sum()),
                                **{metric + "_mean": float(values[:, j].mean()) for j, metric in enumerate(metrics)},
                                **{metric + "_sd": float(values[:, j].std(ddof=1)) for j, metric in enumerate(metrics)}})
    c.write_csv(out / "by_domain_axis.csv", grouped)

    labels = list(recipes)
    colors = ["#6b7280", "#4c78a8", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, j, ylabel, scale in zip(axes, (0, 2, 3), ("H-MAE (meV)", "Band RMSE (meV)", "DOS relative L1 (%)"), (1, 1, 100)):
        means = [arrays[label][:, :, j].mean(axis=(0, 1)) * scale for label in labels]
        sds = [arrays[label][:, :, j].mean(axis=1).std(ddof=1) * scale for label in labels]
        for x, mean, sd, color in zip(range(4), means, sds, colors):
            ax.errorbar(x, mean, yerr=sd, fmt="o", color=color, capsize=4,
                        linewidth=1.5, markersize=6)
        ax.set(xticks=range(4), xticklabels=labels, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("w90, N=64, 4000 epochs; common test (mean ± seed sd, n=5)")
    fig.savefig(out / "dataset_comparison.png", dpi=180)
    fig.savefig(out / "dataset_comparison.pdf")
    plt.close(fig)


def summarize_curve():
    samples = c.read_json(c.OUT / "final_test_manifest.json")["systems"]["w90"]["samples"]
    ids = [s["sample_id"] for s in samples]
    metrics = ("H_MAE_meV", "rel_Frob", "band_rmse_meV", "dos_rel_L1")
    recipe = "random_cartesian__3D__k2__r0.120__res64__scaled_seed0"
    models = [r for r in c.load_results("w90") if r["recipe_id"] == recipe and
              r["stage"] in ("coverage_curve", "coverage_4k")]
    assert len(models) == 25
    arrays, per_seed = {}, []
    for model in models:
        path = c.OUT / "runs/w90" / model["job_id"] / "final_test/per_structure_metrics.csv"
        rows = c.read_csv(path)
        indexed = {r["sample_id"]: r for r in rows}
        assert len(rows) == len(indexed) == len(ids) and set(indexed) == set(ids)
        values = np.array([[float(indexed[sid][metric]) for metric in metrics] for sid in ids])
        arrays[(model["N"], model["training_seed"])] = values
        for domain in ("all", "synthetic", "md"):
            mask = np.array([domain == "all" or s["origin"] == domain for s in samples])
            per_seed.append({"N": model["N"], "seed": model["training_seed"], "domain": domain,
                             "n_structures": int(mask.sum()), "train_minutes": model["train_seconds"] / 60,
                             **dict(zip(metrics, values[mask].mean(axis=0)))})
    cubes = {n: np.stack([arrays[(n, seed)] for seed in range(5)]) for n in c.NS}
    out = c.OUT / "w90_coverage_curve"
    c.write_csv(out / "per_seed.csv", per_seed)
    summary = []
    masks = {"all": np.ones(len(samples), dtype=bool),
             "synthetic": np.array([s["origin"] == "synthetic" for s in samples]),
             "md": np.array([s["origin"] == "md" for s in samples])}
    for n, cube in cubes.items():
        for domain, mask in masks.items():
            values = cube[:, mask].mean(axis=1)
            summary.append({"N": n, "domain": domain, "n_seeds": 5, "n_structures": int(mask.sum()),
                            **{metric + "_mean": float(values[:, j].mean()) for j, metric in enumerate(metrics)},
                            **{metric + "_sd": float(values[:, j].std(ddof=1)) for j, metric in enumerate(metrics)},
                            "train_minutes_mean": float(np.mean([r["train_minutes"] for r in per_seed
                                                                 if r["N"] == n and r["domain"] == domain]))})
    c.write_csv(out / "summary.csv", summary)

    rng = np.random.default_rng(20260922)
    comparisons = []
    for n in (4, 8, 16, 32):
        for domain, mask in masks.items():
            delta = cubes[n][:, mask] - cubes[64][:, mask]
            boot = np.empty((10_000, len(metrics)))
            for i in range(len(boot)):
                seeds = rng.integers(0, 5, 5)
                structures = rng.integers(0, delta.shape[1], delta.shape[1])
                boot[i] = delta[seeds][:, structures].mean(axis=(0, 1))
            for j, metric in enumerate(metrics):
                comparisons.append({"N": n, "reference_N": 64, "domain": domain, "metric": metric,
                                    "mean_delta": float(delta[:, :, j].mean()),
                                    "ci95_low": float(np.percentile(boot[:, j], 2.5)),
                                    "ci95_high": float(np.percentile(boot[:, j], 97.5)),
                                    "better_seeds": int((delta[:, :, j].mean(axis=1) < 0).sum())})
    c.write_csv(out / "paired_vs_N64.csv", comparisons)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, j, ylabel, scale in zip(axes, (0, 2, 3), ("H-MAE (meV)", "Band RMSE (meV)", "DOS relative L1 (%)"), (1, 1, 100)):
        means = [cubes[n][:, :, j].mean() * scale for n in c.NS]
        sds = [cubes[n][:, :, j].mean(axis=1).std(ddof=1) * scale for n in c.NS]
        ax.errorbar(c.NS, means, yerr=sds, fmt="o-", color="#2ca02c", capsize=4)
        ax.set(xscale="log", xticks=c.NS, xticklabels=c.NS, xlabel="Training structures (N)", ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("w90 random 3D R0.12; 8000 updates (mean ± seed sd, n=5)")
    fig.savefig(out / "learning_curve.png", dpi=180)
    fig.savefig(out / "learning_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--curve", action="store_true")
    args = parser.parse_args()
    prepare_pool()
    if args.curve:
        rc = c.cmd_train(argparse.Namespace(systems=["w90"], stage=["coverage_curve"], parallel=10))
        if rc:
            raise SystemExit(rc)
        c.cmd_evaluate_final(argparse.Namespace(systems=["w90"], stages=["coverage_curve", "coverage_4k"], parallel=2))
        summarize_curve()
        print("CURVE COMPLETE: N=4..64, five seeds, summary.csv ready", flush=True)
    elif not args.prepare_only:
        rc = c.cmd_train(argparse.Namespace(systems=["w90"], stage=["coverage_4k"], parallel=3))
        if rc:
            raise SystemExit(rc)
        c.cmd_evaluate_final(argparse.Namespace(systems=["w90"], stages=["coverage_4k", "no_early_stop_4k"], parallel=2))
        summarize()
        print("COVERAGE COMPLETE: 20 models evaluated, summary.csv ready", flush=True)
