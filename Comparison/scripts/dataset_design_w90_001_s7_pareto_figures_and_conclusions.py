#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S7 -- Pareto frontier, 8 figures, 10 conclusions.

Consumes only artifacts already produced by S4/S5/S6 (never re-derives them,
per S1 section 4):
  - ``Comparison/results/dataset_design_w90_001_s4/learning_curves.csv`` (S4):
    per (family, dim, k, N_train, seed) H-MAE/H-RMSE/rel-Frobenius/spectral
    error on the S4 pilot validation split. ``status == "ok"`` rows only are
    used for quantitative comparison; ``insufficient_samples`` rows (every
    ``axial_radial``/``local_pair_modes`` group) are kept in the pruning log
    as data-availability exclusions, never fabricated.
  - ``Comparison/results/dataset_design_w90_001_s5/generalization_matrix.csv``
    and ``per_model_frozen_test_metrics.csv`` (S5): the (training family x
    frozen test) generalization matrix used for the generalization-matrix
    figure and the MD-frozen-test comparison.
  - ``Comparison/results/dataset_design_w90_001_s6/s6_summary.json`` and
    ``non_inferiority_margin.json`` (S6): ``delta_ni_fixed`` gates whether an
    "equivalent to MD" non-inferiority claim may be made. It is ``false`` as
    of this run, so S7 answers conclusion (8) as observed error/cost only,
    per S6's own explicit instruction to any later step.
  - S3/S5 per-sample ``metadata.json`` (radial/angular/locality diagnostics):
    ``amplitude_ang``, ``participation_ratio``, ``radius_of_gyration_ang``.

SIESTA cost convention: one ``Sample`` (S4's dataclass) is materialized from
exactly one SIESTA single-point calculation (S1 section 2), and S4's
``select_split`` draws ``N_train`` samples without replacement from the S3
pool for every (family, dim, k, N_train, seed) point independently (not a
nested/incremental draw -- confirmed by reading ``select_split``). SIESTA
cost for a training point is therefore exactly ``N_train`` unique SIESTA
runs; "best at fixed N" and "best at fixed SIESTA cost" coincide in this
pilot (documented explicitly in conclusion 2 rather than silently assumed).

k>2 / 6x6 caveat: this pilot only ever trained/evaluated k in {1, 2} and 3-5
sampler families (5 defined, 3 with any ``ok`` S4 rows). No figure or
conclusion here extrapolates to k>2 or to a 6x6 family/dimensionality grid;
conclusion 10 states the actually-tested design space explicitly instead.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402

S4_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s4"
S5_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s5"
S6_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s6"
S3_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s3"
OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s7"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
DOCS_PATH = REPO_ROOT / "docs/dataset_design_w90_001_s7_pareto_and_conclusions.md"
LEGACY_RESULTS_ROOT = REPO_ROOT / "Comparison/results"
LEGACY_CONCLUSIONS_PATH = LEGACY_RESULTS_ROOT / "conclusions_w90.json"

OOD_AMPLITUDE_ANG = 0.12
TRAIN_AMPLITUDES_ANG = (0.03, 0.08)
PLATEAU_RELATIVE_GAIN = 0.05  # <5% relative H_MAE improvement counts as "no improvement"

FAMILY_COLORS = {
    "random_cartesian": "#1f77b4",
    "angular_shell": "#ff7f0e",
    "sobol_sparse": "#2ca02c",
    "axial_radial": "#d62728",
    "local_pair_modes": "#9467bd",
}
DIM_ORDER = ["1D_z", "1D_in", "2D_in", "3D"]  # increasing degrees of freedom


# --------------------------------------------------------------------------
# Loading (read-only; never recomputes S4/S5/S6 results)
# --------------------------------------------------------------------------


def load_learning_curves() -> pd.DataFrame:
    df = pd.read_csv(S4_ROOT / "learning_curves.csv")
    for col in ("N_train", "seed", "k"):
        df[col] = df[col].astype(int)
    return df


def load_generalization_matrix() -> pd.DataFrame:
    path = S5_ROOT / "generalization_matrix.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["training_family", "frozen_test", "n_models", "H_MAE_mean", "rel_Frob_mean"]
        )
    return pd.read_csv(path)


def load_per_model_frozen() -> pd.DataFrame:
    path = S5_ROOT / "per_model_frozen_test_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_common_eval() -> pd.DataFrame:
    """Per-(family,dim,k,N_train,seed) H-MAE on S5's ``common_validation`` set.

    ``common_validation`` (S5, distinct Latin-Hypercube seed from
    ``common_displacement``) is the single predeclared validation
    distribution every sampler family is scored on -- as opposed to S4's own
    per-group internal validation split (drawn from that *same family's* S3
    pool), which confounds the ranking with each family's own
    validation-geometry distribution (audit finding: "S7 ranks samplers from
    sampler-specific random validation pools rather than a common ...
    distribution").

    This is deliberately NOT ``common_displacement`` (one of S5's three
    frozen tests): the task rule (S1 section 12) forbids using a frozen test
    to select a sampler/radius/hyperparameter, and doing so would consume
    the one untouched final test for the reported winner (audit finding:
    "no untouched final test remains for the reported winner").
    """

    empty = pd.DataFrame(columns=["family", "dim", "k", "N_train", "seed", "H_MAE"])
    per_model_path = S5_ROOT / "per_model_common_validation_metrics.csv"
    if not per_model_path.exists():
        return empty
    per_model = pd.read_csv(per_model_path)
    common = per_model[per_model["status"] == "ok"].copy()
    if common.empty:
        return empty
    common = common.rename(
        columns={
            "training_family": "family",
            "training_dim": "dim",
            "training_k": "k",
            "training_n_train": "N_train",
            "training_seed": "seed",
        }
    )
    for col in ("k", "N_train", "seed"):
        common[col] = common[col].astype(int)
    return common[["family", "dim", "k", "N_train", "seed", "H_MAE"]]


def load_s6() -> dict[str, Any]:
    summary = json.loads((S6_ROOT / "s6_summary.json").read_text(encoding="utf-8"))
    margin = json.loads((S6_ROOT / "non_inferiority_margin.json").read_text(encoding="utf-8"))
    return {"summary": summary, "margin": margin}


def load_sample_diagnostics() -> pd.DataFrame:
    """Radial/angular/locality metadata for the S3 training pool + S5 frozen tests."""

    rows: list[dict[str, Any]] = []

    def _collect(structures_root: Path, source: str) -> None:
        if not structures_root.exists():
            return
        for struct_dir in sorted(structures_root.iterdir()):
            meta_path = struct_dir / "metadata.json"
            if not struct_dir.is_dir() or not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("family") in (None, "reference"):
                continue
            rows.append(
                {
                    "source": source,
                    "family": meta.get("family"),
                    "dim": meta.get("dimensionality"),
                    "k": meta.get("k"),
                    "amplitude_ang": meta.get("amplitude_ang"),
                    "rms_displacement_ang": meta.get("rms_displacement_ang"),
                    "participation_ratio": meta.get("participation_ratio"),
                    "radius_of_gyration_ang": meta.get("radius_of_gyration_ang"),
                    "neighbor_shell_span": meta.get("neighbor_shell_span"),
                }
            )

    _collect(S3_ROOT / "structures", "s4_train_pool")
    _collect(S5_ROOT / "common_displacement" / "structures", "common_displacement")
    _collect(S5_ROOT / "ood_displacement" / "structures", "ood_displacement")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pareto frontier
# --------------------------------------------------------------------------


def build_pareto_table(df_ok: pd.DataFrame) -> pd.DataFrame:
    """H-MAE vs unique-SIESTA-run cost across every (family, dim, k, N_train, seed) point.

    SIESTA cost is ``N_train + s4.VAL_COUNT``, not ``N_train`` alone: S4's
    nested ``select_split`` now draws one fixed validation pool per
    (family, dim, k, seed) group and reuses it across every nested
    N_train in that curve, so every point still needs the validation
    structures materialized (audit finding: "the stated SIESTA cost of
    exactly N_train ... omits validation structures"). Because train splits
    are nested (train16 subset train32 subset train64), the *marginal* cost
    of the next N_train point is only the newly-added samples -- but the
    absolute unique-run count backing a given point is still
    ``N_train + VAL_COUNT``.
    """

    table = df_ok[["family", "dim", "k", "N_train", "seed", "H_MAE"]].copy()
    table["siesta_cost"] = table["N_train"] + s4.VAL_COUNT
    table = table.drop(columns=["N_train"])
    table = table.sort_values(["siesta_cost", "H_MAE"]).reset_index(drop=True)

    best_error_so_far = np.inf
    is_frontier = []
    for _, row in table.iterrows():
        if row["H_MAE"] < best_error_so_far:
            is_frontier.append(True)
            best_error_so_far = row["H_MAE"]
        else:
            is_frontier.append(False)
    table["pareto_dominated"] = ~np.array(is_frontier)
    return table


# --------------------------------------------------------------------------
# Pruning log
# --------------------------------------------------------------------------


def build_pruning_log(df_all: pd.DataFrame, pareto_table: pd.DataFrame) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []

    # (1) Data-availability exclusions: families with zero "ok" rows anywhere.
    for family in sorted(df_all["family"].unique()):
        family_rows = df_all[df_all["family"] == family]
        if (family_rows["status"] == "ok").sum() == 0:
            reasons = sorted(family_rows["status"].unique())
            log.append(
                {
                    "scope": f"family={family}",
                    "decision": "excluded_pre_pareto",
                    "rule": "data_unavailable",
                    "rationale": (
                        f"Every (dim, k, N_train, seed) point for family={family!r} is "
                        f"status in {reasons} in S4's learning_curves.csv (0 ok rows). "
                        "S4's docstring already attributes this to too few S3 samples "
                        "for this deterministic geometric family, not to model quality; "
                        "excluded from the Pareto frontier and every quantitative figure."
                    ),
                }
            )

    ok_families = sorted(df_all.loc[df_all["status"] == "ok", "family"].unique())

    # (2) Pareto-dominated strategies: never on the frontier at any cost.
    frontier_keys = set(
        tuple(r) for r in pareto_table.loc[~pareto_table["pareto_dominated"], ["family", "dim", "k"]].itertuples(index=False)
    )
    for family in ok_families:
        for dim in DIM_ORDER:
            for k in (1, 2):
                subset = pareto_table[(pareto_table.family == family) & (pareto_table.dim == dim) & (pareto_table.k == k)]
                if subset.empty:
                    continue
                on_frontier = (family, dim, k) in frontier_keys
                if not on_frontier:
                    log.append(
                        {
                            "scope": f"family={family}, dim={dim}, k={k}",
                            "decision": "pareto_dominated",
                            "rule": "never_on_frontier",
                            "rationale": (
                                f"No (N_train, seed) point for this (family, dim, k) group has the "
                                f"minimum H_MAE at its SIESTA cost; every point is strictly beaten by "
                                f"a cheaper-or-equal-cost point from another strategy "
                                f"(min H_MAE observed here: {subset['H_MAE'].min():.4f} eV)."
                            ),
                        }
                    )

    # (3) No improvement over two N increments (16->32 and 32->64).
    for family in ok_families:
        for dim in DIM_ORDER:
            for k in (1, 2):
                subset = df_all[
                    (df_all.family == family) & (df_all.dim == dim) & (df_all.k == k) & (df_all.status == "ok")
                ]
                if subset.empty:
                    continue
                by_n = subset.groupby("N_train")["H_MAE"].mean().sort_index()
                if list(by_n.index) != [16, 32, 64]:
                    continue
                gain_16_32 = (by_n[16] - by_n[32]) / by_n[16]
                gain_32_64 = (by_n[32] - by_n[64]) / by_n[32]
                if gain_16_32 < PLATEAU_RELATIVE_GAIN and gain_32_64 < PLATEAU_RELATIVE_GAIN:
                    log.append(
                        {
                            "scope": f"family={family}, dim={dim}, k={k}",
                            "decision": "flagged_saturated",
                            "rule": "no_improvement_two_N_increments",
                            "rationale": (
                                f"Mean H_MAE barely moves across both nested increments: "
                                f"N=16->32 gain={gain_16_32:.1%}, N=32->64 gain={gain_32_64:.1%} "
                                f"(both below the {PLATEAU_RELATIVE_GAIN:.0%} threshold); "
                                f"H_MAE(16,32,64) = ({by_n[16]:.4f}, {by_n[32]:.4f}, {by_n[64]:.4f}) eV. "
                                "Not pruned outright (still informs the saturation-point conclusion), "
                                "but additional N_train beyond 64 is not justified by this evidence."
                            ),
                        }
                    )

    return log


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, "#7f7f7f")


def dimensionality_common_dims(df_ok: pd.DataFrame) -> list[str]:
    """Dims (in ``DIM_ORDER``) that every family with any ``ok`` row also has data for.

    Pooling H-MAE across a dim that only some families cover (e.g.
    ``angular_shell`` never ran 1D) confounds "dimensionality effect" with
    "which families happen to be present at that dim" -- audit finding:
    "the pooled dimensionality means ... cannot quantify the effect of
    dimensionality". Restricting to the intersection keeps the comparison
    apples-to-apples, at the cost of only speaking to the dims every
    surviving family actually covers.
    """

    families = sorted(df_ok["family"].unique())
    if not families:
        return []
    return [
        dim
        for dim in DIM_ORDER
        if all(not df_ok[(df_ok.family == f) & (df_ok.dim == dim)].empty for f in families)
    ]


def fig1_learning_curves(df_ok: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for family, group in df_ok.groupby("family"):
        stats = group.groupby("N_train")["H_MAE"].agg(["mean", "std"]).sort_index()
        ax.errorbar(
            stats.index, stats["mean"], yerr=stats["std"].fillna(0.0),
            marker="o", label=family, color=_family_color(family), capsize=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(["16", "32", "64"])
    ax.set_xlabel("N_train (samples)")
    ax.set_ylabel("H-MAE (eV)")
    ax.set_title("S4 learning curves: H-MAE vs N_train, by sampler family\n(mean +/- std over dim, k, seed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig2_dft_efficiency(df_ok: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for family, group in df_ok.groupby("family"):
        ax.scatter(group["N_train"], group["H_MAE"], color=_family_color(family), label=family, alpha=0.6, s=25)
    ax.set_xlabel("Unique SIESTA runs required (= N_train, S1 convention)")
    ax.set_ylabel("H-MAE (eV)")
    ax.set_title("DFT efficiency: H-MAE per unique SIESTA calculation spent\n(every (family, dim, k, seed) point)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig3_pareto(pareto_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    dominated = pareto_table[pareto_table.pareto_dominated]
    frontier = pareto_table[~pareto_table.pareto_dominated].sort_values("siesta_cost")
    ax.scatter(dominated["siesta_cost"], dominated["H_MAE"], color="#cccccc", s=20, label="dominated")
    ax.plot(frontier["siesta_cost"], frontier["H_MAE"], "o-", color="#d62728", label="Pareto frontier")
    for _, row in frontier.iterrows():
        ax.annotate(f"{row['family']}/{row['dim']}/k{row['k']}", (row["siesta_cost"], row["H_MAE"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("SIESTA cost (unique runs)")
    ax.set_ylabel("H-MAE (eV)")
    ax.set_title("Pareto frontier: H-MAE vs SIESTA cost, all strategies")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig4_generalization_matrix(matrix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if matrix.empty:
        ax.text(0.5, 0.5, "S5 generalization matrix not available", ha="center", va="center")
        ax.axis("off")
    else:
        families = sorted(matrix["training_family"].unique())
        tests = sorted(matrix["frozen_test"].unique())
        grid = np.full((len(families), len(tests)), np.nan)
        for _, row in matrix.iterrows():
            i = families.index(row["training_family"])
            j = tests.index(row["frozen_test"])
            grid[i, j] = row["H_MAE_mean"]
        im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(tests)), tests, rotation=20, ha="right")
        ax.set_yticks(range(len(families)), families)
        for i in range(len(families)):
            for j in range(len(tests)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, label="H-MAE mean (eV)")
    ax.set_title("Generalization matrix: training family x frozen test (S5)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig5_collectivity_locality(df_ok: pd.DataFrame, diagnostics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    by_k = df_ok.groupby(["family", "k"])["H_MAE"].mean().reset_index()
    width = 0.35
    families = sorted(by_k["family"].unique())
    x = np.arange(len(families))
    for offset, k in zip((-width / 2, width / 2), (1, 2)):
        values = [by_k[(by_k.family == f) & (by_k.k == k)]["H_MAE"].mean() for f in families]
        ax.bar(x + offset, values, width, label=f"k={k}")
    ax.set_xticks(x, families, rotation=20, ha="right")
    ax.set_ylabel("H-MAE (eV)")
    ax.set_title("Collectivity: H-MAE by k (displaced-atom count)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    if not diagnostics.empty:
        pool = diagnostics[diagnostics.source == "s4_train_pool"]
        for family, group in pool.groupby("family"):
            ax.scatter(group["radius_of_gyration_ang"], group["participation_ratio"],
                       color=_family_color(family), label=family, alpha=0.5, s=15)
    ax.set_xlabel("Radius of gyration of displacement (Ang) -- locality")
    ax.set_ylabel("Participation ratio -- collectivity")
    ax.set_title("Locality vs collectivity coverage (S3 training pool)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig6_dimensionality_comparison(df_ok: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    families = sorted(df_ok["family"].unique())
    x = np.arange(len(DIM_ORDER))
    width = 0.8 / max(len(families), 1)
    for idx, family in enumerate(families):
        values = []
        for dim in DIM_ORDER:
            subset = df_ok[(df_ok.family == family) & (df_ok.dim == dim)]
            values.append(subset["H_MAE"].mean() if not subset.empty else np.nan)
        ax.bar(x + idx * width, values, width, label=family, color=_family_color(family))
    ax.set_xticks(x + width * (len(families) - 1) / 2, DIM_ORDER)
    ax.set_xlabel("Displacement dimensionality (1D_z -> 1D_in -> 2D_in -> 3D)")
    ax.set_ylabel("H-MAE (eV)")
    ax.set_title("Dimensionality comparison: H-MAE by 1D/2D/3D displacement space\n(per family -- bars are NOT pooled across families with unequal dim coverage)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    excluded_dims = [d for d in DIM_ORDER if d not in dimensionality_common_dims(df_ok)]
    if excluded_dims:
        fig.text(
            0.02, 0.02,
            f"Dims not covered by every family (excluded from the pooled dimensionality "
            f"conclusion): {', '.join(excluded_dims)}",
            fontsize=7, style="italic",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig7_sampled_space_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    sources = ["s4_train_pool", "common_displacement", "ood_displacement"]
    labels = {"s4_train_pool": "S4 train pool", "common_displacement": "S5 common (frozen)", "ood_displacement": "S5 OOD (frozen)"}

    ax = axes[0]
    for source in sources:
        values = diagnostics.loc[diagnostics.source == source, "amplitude_ang"].dropna()
        if len(values):
            ax.hist(values, bins=20, alpha=0.5, label=labels[source])
    ax.axvline(OOD_AMPLITUDE_ANG, color="black", linestyle="--", linewidth=1, label="OOD amplitude (0.12 Ang)")
    ax.set_xlabel("amplitude_ang (radial coverage)")
    ax.set_ylabel("count")
    ax.set_title("Radial coverage")
    ax.legend(fontsize=6)

    ax = axes[1]
    for source in sources:
        values = diagnostics.loc[diagnostics.source == source, "participation_ratio"].dropna()
        if len(values):
            ax.hist(values, bins=20, alpha=0.5, label=labels[source])
    ax.set_xlabel("participation_ratio (angular/collectivity coverage)")
    ax.set_title("Angular coverage")
    ax.legend(fontsize=6)

    ax = axes[2]
    for source in sources:
        values = diagnostics.loc[diagnostics.source == source, "radius_of_gyration_ang"].dropna()
        if len(values):
            ax.hist(values, bins=20, alpha=0.5, label=labels[source])
    ax.set_xlabel("radius_of_gyration_ang (locality coverage)")
    ax.set_title("Locality coverage")
    ax.legend(fontsize=6)

    fig.suptitle("Sampled-space diagnostics: train pool vs frozen tests")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig8_designed_vs_md(matrix: pd.DataFrame, s6: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    md_rows = matrix[matrix.frozen_test == "md_frozen"].sort_values("H_MAE_mean")
    if md_rows.empty:
        ax.text(0.5, 0.6, "No S5 md_frozen evaluation available", ha="center", va="center")
    else:
        colors = [_family_color(f) for f in md_rows["training_family"]]
        ax.bar(md_rows["training_family"], md_rows["H_MAE_mean"], color=colors)
        ax.set_ylabel("H-MAE on md_frozen (eV, S5)")
        ax.set_xlabel("Training family (designed displacement dataset)")
        ax.tick_params(axis="x", rotation=20)
    delta_ni_fixed = s6["margin"]["delta_ni_fixed"]
    note = (
        "delta_NI NOT fixed (S6): no non-inferiority claim is made here -- observed\n"
        "error/cost only. MD training baseline itself could not be trained (S6: 13\n"
        "available frames, insufficient for N_train>=16)."
        if not delta_ni_fixed
        else f"delta_NI = {s6['margin']['delta_ni_relative_h_mae']}"
    )
    ax.set_title("Designed datasets vs MD: observed error on MD-frozen test\n(no MD-trained baseline exists to compare against)")
    fig.text(0.02, 0.02, note, fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Conclusions
# --------------------------------------------------------------------------


def answer_conclusions(
    df_all: pd.DataFrame,
    df_ok: pd.DataFrame,
    pareto_table: pd.DataFrame,
    matrix: pd.DataFrame,
    s6: dict[str, Any],
    df_rank: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# DATASET-DESIGN-W90-001-S7 -- Pareto frontier, figures, conclusions\n")
    lines.append(
        "Cites `docs/dataset_design_w90_001_s1_convention_manifest.md` (S1), and reads "
        "S4/S5/S6 outputs verbatim (`Comparison/results/dataset_design_w90_001_s{4,5,6}/`), "
        "never recomputing them. Produced by "
        "`Comparison/scripts/dataset_design_w90_001_s7_pareto_figures_and_conclusions.py`.\n"
    )

    using_common_eval = not df_rank.empty
    rank_source = (
        "S5's `common_validation` set (one predeclared validation distribution shared by every "
        "training family, disjoint from S5's three frozen tests -- never used to name a winner "
        "from a frozen test, per S1 section 12)"
        if using_common_eval
        else "S4's own per-group internal validation split (fallback: S5 `common_validation` "
        "evaluations are not available yet, so this ranking is confounded by each family's own "
        "validation-geometry distribution and should be treated as provisional)"
    )

    # 1. Best sampler at fixed N
    lines.append("## 1. Best sampler at fixed N_train\n")
    lines.append(f"Ranking source: {rank_source}.\n")
    for n_train in (16, 32, 64):
        sub = df_rank[df_rank.N_train == n_train].groupby("family")["H_MAE"].mean().sort_values()
        if sub.empty:
            lines.append(f"- N_train={n_train}: no ranking data available.")
            continue
        best = sub.index[0]
        lines.append(f"- N_train={n_train}: best family = **{best}** (mean H-MAE = {sub.iloc[0]:.4f} eV); "
                      f"full ranking: {', '.join(f'{f}={v:.4f}' for f, v in sub.items())}.")
    lines.append("")

    # 2. Best at fixed SIESTA cost
    lines.append("## 2. Best sampler at fixed SIESTA cost\n")
    lines.append(
        f"Ranking source: {rank_source}. SIESTA cost for a training point is "
        "`N_train + s4.VAL_COUNT` unique SIESTA single-point calculations (S4's `select_split` "
        "now draws one *fixed* validation pool per (family, dim, k, seed) group, reused -- not "
        "re-drawn -- across every nested N_train, so `train16 subset train32 subset train64`; "
        "the marginal cost of the next N_train point is only its newly-added samples, but the "
        "absolute unique-run count backing a point still includes the shared validation pool). "
        "Because every family pays the same fixed VAL_COUNT offset, the fixed-cost ranking at "
        "cost=N_train+VAL_COUNT is identical to conclusion 1's fixed-N ranking; no separate "
        "cost-sharing effect was observed or assumed.\n"
    )

    # 3. 1D -> 2D -> 3D contribution
    lines.append("## 3. Contribution of 1D -> 2D -> 3D displacement space\n")
    common_dims = dimensionality_common_dims(df_ok)
    excluded_dims = [d for d in DIM_ORDER if d not in common_dims]
    if len(common_dims) < 2:
        lines.append(
            "No two dimensionalities have `ok` S4 data from every family with any `ok` data at "
            "all -- pooling across families with unequal dimensionality availability would "
            "confound family choice with dimensionality (audit finding: e.g. `angular_shell` has "
            "no 1D samples). No 1D->2D->3D trend is claimed.\n"
        )
    else:
        df_common_dim = df_ok[df_ok["dim"].isin(common_dims)]
        by_dim = df_common_dim.groupby("dim")["H_MAE"].mean().reindex(common_dims)
        lines.append(
            f"Mean H-MAE by dimensionality, restricted to dims where *every* family with any `ok` "
            f"S4 data also has data ({', '.join(common_dims)}); "
            + (
                f"excluded from this pooling because coverage is not shared across families: "
                f"{', '.join(excluded_dims)}.\n"
                if excluded_dims
                else "(every tested dimensionality is covered by every family).\n"
            )
        )
        for dim, val in by_dim.items():
            if pd.notna(val):
                lines.append(f"- {dim}: {val:.4f} eV")
        trend = "increases" if by_dim.dropna().iloc[-1] > by_dim.dropna().iloc[0] else "decreases"
        lines.append(f"\nMean H-MAE {trend} from {by_dim.dropna().index[0]} to {by_dim.dropna().index[-1]} "
                     f"across the dims common to every family; see `fig6_dimensionality_comparison.png` for "
                     f"the per-family breakdown (the pooled trend is not necessarily monotonic per family). "
                     f"This does not quantify {', '.join(excluded_dims)} coverage, which only exists for a "
                     f"subset of families.\n" if excluded_dims else "\n")

    # 4. k=2 vs k=1
    lines.append("## 4. k=2 vs k=1\n")
    by_k = df_ok.groupby("k")["H_MAE"].mean()
    lines.append(f"Mean H-MAE: k=1 -> {by_k.get(1, float('nan')):.4f} eV, k=2 -> {by_k.get(2, float('nan')):.4f} eV "
                 f"(pooled over family, dim, N_train, seed; see `fig5_collectivity_locality.png` for the "
                 f"per-family split). This pilot never trained or evaluated k>2 -- no claim is made "
                 f"beyond k in {{1, 2}}.\n")

    # 5. Required amplitudes
    lines.append("## 5. Required displacement amplitudes\n")
    common_row = matrix[matrix.frozen_test == "common_displacement"]
    ood_row = matrix[matrix.frozen_test == "ood_displacement"]
    if not common_row.empty and not ood_row.empty:
        c_mean = common_row["H_MAE_mean"].mean()
        o_mean = ood_row["H_MAE_mean"].mean()
        lines.append(
            f"S3/S4 trained only at amplitudes {TRAIN_AMPLITUDES_ANG} Ang. S5's `common_displacement` "
            f"frozen test (amplitudes drawn continuously within that same 0.03-0.08 Ang range) gives "
            f"mean H-MAE {c_mean:.4f} eV across training families, vs {o_mean:.4f} eV on `ood_displacement` "
            f"(amplitude {OOD_AMPLITUDE_ANG} Ang, outside the training grid). "
            f"{'Error grows outside the trained amplitude range' if o_mean > c_mean else 'Error does not clearly grow outside the trained amplitude range in this pilot'}, "
            f"i.e. amplitude coverage up to the intended deployment displacement scale is required -- "
            f"0.03-0.08 Ang alone does not certify accuracy at 0.12 Ang."
        )
    else:
        lines.append("S5 generalization matrix unavailable for common_displacement/ood_displacement; "
                     "no amplitude-extrapolation claim is made.")
    lines.append("")

    # 6. Saturation point
    lines.append("## 6. Saturation point\n")
    saturated = [row for row in build_pruning_log(df_all, pareto_table) if row["rule"] == "no_improvement_two_N_increments"]
    if saturated:
        for row in saturated:
            lines.append(f"- {row['scope']}: {row['rationale']}")
    else:
        lines.append("No (family, dim, k) group with full N_train in {16,32,64} coverage shows "
                     f"<{PLATEAU_RELATIVE_GAIN:.0%} relative H-MAE improvement across both N increments; "
                     "no saturation is claimed within the tested N_train <= 64 range.")
    lines.append("")

    # 7. Generalization to MD
    lines.append("## 7. Generalization to real MD frames\n")
    md_rows = matrix[matrix.frozen_test == "md_frozen"].sort_values("H_MAE_mean")
    if not md_rows.empty:
        n_models = int(md_rows["n_models"].sum())
        lines.append(
            f"Every S4 model was evaluated on the 7-frame `md_frozen` decorrelated MD test (S5). "
            f"Best-generalizing training family: **{md_rows.iloc[0]['training_family']}** "
            f"(mean H-MAE = {md_rows.iloc[0]['H_MAE_mean']:.4f} eV over {md_rows.iloc[0]['n_models']} models); "
            f"worst: **{md_rows.iloc[-1]['training_family']}** (mean H-MAE = {md_rows.iloc[-1]['H_MAE_mean']:.4f} eV). "
            f"Caveat: only 7 MD frames are available (S6), all from one trajectory region -- this is a "
            f"small-sample, single-trajectory generalization estimate, not a paper-grade MD test."
        )
    else:
        lines.append("S5 md_frozen evaluation unavailable.")
    lines.append("")

    # 8. MD-replacement claim
    lines.append("## 8. Can a designed displacement dataset replace MD training data?\n")
    if not s6["margin"]["delta_ni_fixed"]:
        lines.append(
            "**No non-inferiority claim is made.** S6 explicitly could not fix a non-inferiority "
            "margin (`delta_ni_fixed: false` in `non_inferiority_margin.json`): no MD-trained "
            "baseline model exists at all (every N_train in {16,32,64} is `insufficient_md_training_data`, "
            "13 available MD training frames vs >=16 needed), so there is nothing to be non-inferior "
            "to. Per S6's own instruction to later steps, this conclusion is limited to the observed "
            "error/cost and Pareto front above (sections 1-4, 9): designed displacement samplers reach "
            f"H-MAE {df_ok['H_MAE'].min():.4f}-{df_ok['H_MAE'].max():.4f} eV at N_train in {{16,32,64}} "
            "on their own validation split, and generalize to `md_frozen` with the errors reported in "
            "section 7 -- but whether that is 'as good as MD training data' is undefined without an MD "
            "baseline to compare against."
        )
    else:
        lines.append(f"delta_NI = {s6['margin']['delta_ni_relative_h_mae']} (fixed by S6); apply directly.")
    lines.append("")

    # 9. Pareto set
    lines.append("## 9. Pareto-optimal set\n")
    frontier = pareto_table[~pareto_table.pareto_dominated].sort_values("siesta_cost")
    lines.append("Non-dominated (family, dim, k, N_train, seed) points (H-MAE vs SIESTA-run cost):\n")
    for _, row in frontier.iterrows():
        lines.append(f"- cost={row['siesta_cost']}: {row['family']}/{row['dim']}/k={row['k']}/seed={row['seed']}, "
                     f"H-MAE={row['H_MAE']:.4f} eV")
    lines.append(f"\nFull table: `pareto_table.csv` ({len(pareto_table)} points, {len(frontier)} on the frontier).\n")

    # 10. 6x6 candidates / k>2 caveat
    lines.append("## 10. 6x6 candidate grid / k>2 extrapolation\n")
    n_families_defined = df_all["family"].nunique()
    n_families_ok = df_ok["family"].nunique()
    lines.append(
        f"This pilot defined {n_families_defined} sampler families "
        f"({', '.join(sorted(df_all['family'].unique()))}), of which only {n_families_ok} "
        f"({', '.join(sorted(df_ok['family'].unique()))}) produced any `ok` S4 training point -- "
        f"`axial_radial` and `local_pair_modes` never had enough S3 samples (see pruning log). "
        f"Coverage actually exercised: {n_families_ok} usable families x {len(DIM_ORDER)} dimensionalities "
        f"x 2 k values (1, 2) x 3 N_train x 2 seeds. **No 6x6 family/dimensionality grid was "
        "constructed, and k was never trained or evaluated above 2.** Every conclusion and figure "
        "above is scoped to this tested space; extending any of them to a 6x6 grid or to k>2 would "
        "be extrapolation beyond the evidence and is explicitly not done here."
    )
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> int:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    df_all = load_learning_curves()
    df_ok = df_all[df_all.status == "ok"].copy()
    matrix = load_generalization_matrix()
    per_model = load_per_model_frozen()
    s6 = load_s6()
    diagnostics = load_sample_diagnostics()

    # "Best sampler" ranking/Pareto must come from a validation distribution
    # shared by every family, not each family's own internal S4 split
    # (audit finding on distribution confounding) -- fall back to the
    # internal split only if S5's common_displacement evaluation is missing.
    df_common = load_common_eval()
    df_rank = df_common if not df_common.empty else df_ok

    pareto_table = build_pareto_table(df_rank)
    pruning_log = build_pruning_log(df_all, pareto_table)

    fig1_learning_curves(df_rank, FIGURES_ROOT / "fig1_learning_curves.png")
    fig2_dft_efficiency(df_rank, FIGURES_ROOT / "fig2_dft_efficiency.png")
    fig3_pareto(pareto_table, FIGURES_ROOT / "fig3_pareto_frontier.png")
    fig4_generalization_matrix(matrix, FIGURES_ROOT / "fig4_generalization_matrix.png")
    fig5_collectivity_locality(df_ok, diagnostics, FIGURES_ROOT / "fig5_collectivity_locality.png")
    fig6_dimensionality_comparison(df_ok, FIGURES_ROOT / "fig6_dimensionality_comparison.png")
    fig7_sampled_space_diagnostics(diagnostics, FIGURES_ROOT / "fig7_sampled_space_diagnostics.png")
    fig8_designed_vs_md(matrix, s6, FIGURES_ROOT / "fig8_designed_vs_md.png")

    # Keep the task validator's legacy flat output contract in sync with the
    # canonical S7 directory without generating a second set of figures.
    for figure in FIGURES_ROOT.glob("*.png"):
        shutil.copy2(figure, LEGACY_RESULTS_ROOT / figure.name)

    pareto_table.to_csv(OUTPUT_ROOT / "pareto_table.csv", index=False)
    (OUTPUT_ROOT / "pruning_log.json").write_text(
        json.dumps(pruning_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    conclusions_md = answer_conclusions(df_all, df_ok, pareto_table, matrix, s6, df_rank)
    DOCS_PATH.write_text(conclusions_md, encoding="utf-8")
    LEGACY_CONCLUSIONS_PATH.write_text(
        json.dumps(
            {
                "task": "DATASET-DESIGN-W90-001-S7",
                "document": str(DOCS_PATH.relative_to(REPO_ROOT)),
                "markdown": conclusions_md,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "task": "DATASET-DESIGN-W90-001-S7",
        "ok_rows": int(len(df_ok)),
        "total_rows": int(len(df_all)),
        "pareto_frontier_points": int((~pareto_table["pareto_dominated"]).sum()),
        "pruning_log_entries": len(pruning_log),
        "families_with_data": sorted(df_ok["family"].unique().tolist()),
        "families_excluded": sorted(set(df_all["family"].unique()) - set(df_ok["family"].unique())),
        "delta_ni_fixed": s6["margin"]["delta_ni_fixed"],
        "generalization_matrix_rows": int(len(matrix)),
        "figures": sorted(p.name for p in FIGURES_ROOT.glob("*.png")),
    }
    (OUTPUT_ROOT / "s7_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
