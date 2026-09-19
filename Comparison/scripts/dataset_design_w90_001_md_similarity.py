#!/usr/bin/env python3
"""How MD-like is each S4-pilot sampler family (graphene 5x5)?

Two metrics only (deliberately not the full battery a literature review
suggests -- see the task's own instruction to keep this simple):

  D_amp  -- Wasserstein-1 distance between the pooled |displacement| of a
            family's ACTIVE atoms (every atom it ever moves, across every
            pilot design of that family) and real MD's per-atom |u|,
            normalized by MD's RMS. Only active atoms count: most S4-pilot
            designs move just k=1 or k=2 atoms per config, so including the
            other ~48 always-zero atoms would bias the synthetic
            distribution toward zero for no physical reason.

  D_corr -- RMSE, over the first 3 graphene neighbor shells, between real
            MD's shell-averaged displacement correlation C(shell) and each
            family's own active-PAIR correlation at that shell (from its
            k=2 designs only -- k=1 designs move a single atom and have no
            pair to correlate by construction).

# ponytail: no S(q) spectrum, no PCA+MMD, no 0-1 calibration against
# MD-vs-shuffled-MD, no temperature split, no bootstrap. If D_amp/D_corr
# already separate the families, that's enough for a first answer; add the
# rest only if this plot turns out ambiguous.

Real MD reference: 1000 SIESTA-relaxed-then-thermalized snapshots at
Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing/
graphene_5x5_scale_iid1000/MD_steps/<i>/graphene_5x5.XV, displacement taken
against materials/graphene_5x5/RUN.fdf (the same reference
w90_displacement_sampler_family.load_graphene_5x5() uses, so synthetic and MD
displacements are defined identically). No MD reference exists for the 6x6
supercell (that geometry has no MD trajectory in this repo), so this
analysis only covers the 6 S4-pilot sampler families, not the w90/6x6
cross-testing campaign.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import w90_displacement_sampler_family as sampler  # noqa: E402
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402

BOHR_TO_ANG = 0.529177210903
MD_STEPS_ROOT = (
    REPO_ROOT / "Comparison/datasets/graphene_5x5_snapshot_scaling_complete_paper_ready_missing"
    "/graphene_5x5_scale_iid1000/MD_steps"
)
RUN_METRICS_PATH = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s4/run_metrics.csv"
OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_md_similarity"
N_SHELLS = 3
SHELL_TOL_ANG = 0.15
FAMILY_COLORS = {
    "random_cartesian": "#1f77b4", "angular_shell": "#ff7f0e", "sobol_sparse": "#2ca02c",
    "axial_radial": "#d62728", "local_pair_modes": "#9467bd", "latin_hypercube": "#8c564b",
}


def parse_xv_positions_ang(xv_path: Path, n_atoms: int) -> np.ndarray:
    lines = [line for line in xv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    positions = np.zeros((n_atoms, 3))
    for i, line in enumerate(lines[4 : 4 + n_atoms]):
        parts = line.split()
        positions[i] = [float(v) * BOHR_TO_ANG for v in parts[2:5]]
    return positions


def load_md_displacements(geometry: sampler.Geometry) -> np.ndarray:
    ref = geometry.positions_ang
    xv_files = sorted(MD_STEPS_ROOT.glob("*/graphene_5x5.XV"), key=lambda p: int(p.parent.name))
    disps = np.empty((len(xv_files), geometry.n_atoms, 3))
    for s_idx, xv in enumerate(xv_files):
        pos = parse_xv_positions_ang(xv, geometry.n_atoms)
        u = pos - ref
        u -= u.mean(axis=0)  # drop rigid-translation drift
        disps[s_idx] = u
    return disps  # (S, N, 3)


def neighbor_shells(geometry: sampler.Geometry) -> dict[frozenset[int], int]:
    """(i,j) pair -> shell index (0,1,2) for the first N_SHELLS distinct graphene neighbor distances."""
    n = geometry.n_atoms
    pair_dist = []
    for i in range(n):
        for j in range(i + 1, n):
            _, d = sampler.min_image_vector(geometry, i, j)
            pair_dist.append((d, i, j))
    pair_dist.sort(key=lambda t: t[0])
    radii: list[float] = []
    for d, _, _ in pair_dist:
        if len(radii) >= N_SHELLS:
            break
        if not radii or all(abs(d - r) > SHELL_TOL_ANG for r in radii):
            radii.append(d)
    shell_of: dict[frozenset[int], int] = {}
    for d, i, j in pair_dist:
        for shell_idx, r in enumerate(radii):
            if abs(d - r) <= SHELL_TOL_ANG:
                shell_of[frozenset((i, j))] = shell_idx
                break
    return shell_of, radii


def md_shell_correlation(md_disps: np.ndarray, shell_of: dict[frozenset[int], int]) -> dict[int, float]:
    by_shell: dict[int, list[float]] = defaultdict(list)
    for pair, shell in shell_of.items():
        i, j = tuple(pair)
        ui, uj = md_disps[:, i, :], md_disps[:, j, :]
        denom = np.sqrt((ui**2).sum(axis=1).mean() * (uj**2).sum(axis=1).mean())
        if denom > 0:
            by_shell[shell].append(float(np.sum(ui * uj, axis=1).mean() / denom))
    return {shell: float(np.mean(vals)) for shell, vals in by_shell.items() if vals}


def load_pilot_rows() -> list[dict[str, str]]:
    with RUN_METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if r["status"] == "ok" and r["test_amplitude"] == "development_set"]


def family_active_displacements(geometry: sampler.Geometry, rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[frozenset[int], list[float]]]:
    """Every active-atom |u| (for D_amp) and every k=2 active-pair correlation (for D_corr), pooled over every unique pilot design of this family."""
    magnitudes: list[float] = []
    pair_corr: dict[frozenset[int], list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["design_id"], row["seed"])
        if key in seen:
            continue
        seen.add(key)
        k = int(row["k"])
        configs = s3.generate_design(
            geometry, row["family"], row["dim"], k, float(row["domain"]), int(row["density"]), int(row["seed"]),
        )
        for cfg in configs:
            active = list(cfg.displacements_ang.items())
            for _, vec in active:
                magnitudes.append(float(np.linalg.norm(vec)))
            if len(active) == 2:
                (i, ui), (j, uj) = active
                ui_arr, uj_arr = np.array(ui), np.array(uj)
                denom = np.linalg.norm(ui_arr) * np.linalg.norm(uj_arr)
                if denom > 0:
                    pair_corr[frozenset((i, j))].append(float(np.dot(ui_arr, uj_arr) / denom))
    return np.array(magnitudes), pair_corr


def family_shell_correlation(pair_corr: dict[frozenset[int], list[float]], shell_of: dict[frozenset[int], int]) -> dict[int, float]:
    by_shell: dict[int, list[float]] = defaultdict(list)
    for pair, values in pair_corr.items():
        shell = shell_of.get(pair)
        if shell is not None:
            by_shell[shell].extend(values)
    return {shell: float(np.mean(vals)) for shell, vals in by_shell.items() if vals}


def main() -> int:
    geometry = sampler.load_graphene_5x5()
    print(json.dumps({"phase": "loading_md", "n_atoms": geometry.n_atoms}), flush=True)
    md_disps = load_md_displacements(geometry)
    md_magnitudes = np.linalg.norm(md_disps.reshape(-1, 3), axis=1)
    md_rms = float(np.sqrt(np.mean(md_magnitudes**2)))
    shell_of, shell_radii_ang = neighbor_shells(geometry)
    md_corr = md_shell_correlation(md_disps, shell_of)
    print(json.dumps({"phase": "md_ready", "n_snapshots": len(md_disps), "md_rms_ang": md_rms,
                       "shell_radii_ang": shell_radii_ang, "md_corr_by_shell": md_corr}, indent=2), flush=True)

    pilot_rows = load_pilot_rows()
    families = sorted({r["family"] for r in pilot_rows})
    h_mae_by_family = defaultdict(list)
    rel_frob_by_family = defaultdict(list)
    for r in pilot_rows:
        h_mae_by_family[r["family"]].append(float(r["H_MAE"]))
        if r.get("rel_Frob"):
            rel_frob_by_family[r["family"]].append(float(r["rel_Frob"]))

    results: list[dict[str, Any]] = []
    for family in families:
        rows = [r for r in pilot_rows if r["family"] == family]
        magnitudes, pair_corr = family_active_displacements(geometry, rows)
        d_amp = float(wasserstein_distance(magnitudes, md_magnitudes) / md_rms)
        synth_corr = family_shell_correlation(pair_corr, shell_of)
        shells_in_common = sorted(set(synth_corr) & set(md_corr))
        d_corr = (
            float(np.sqrt(np.mean([(synth_corr[s] - md_corr[s]) ** 2 for s in shells_in_common])))
            if shells_in_common else None
        )
        result = {
            "family": family, "D_amp": d_amp, "D_corr": d_corr,
            "n_configs_pooled": int(len(magnitudes)), "n_k2_pairs": int(len(pair_corr)),
            "H_MAE_mean": float(np.mean(h_mae_by_family[family])),
            "H_MAE_n": len(h_mae_by_family[family]),
            "rel_Frob_mean": float(np.mean(rel_frob_by_family[family])) if rel_frob_by_family[family] else None,
            "rel_Frob_n": len(rel_frob_by_family[family]),
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "md_similarity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    fig_a, ax_a = plt.subplots(figsize=(6, 5))
    for r in results:
        if r["D_corr"] is None:
            continue
        ax_a.scatter(r["D_amp"], r["D_corr"], s=80, color=FAMILY_COLORS.get(r["family"], "#000"), label=r["family"])
        ax_a.annotate(r["family"], (r["D_amp"], r["D_corr"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax_a.set_xlabel("D_amp (Wasserstein-1 / MD RMS)")
    ax_a.set_ylabel("D_corr (RMSE, shell correlation vs MD)")
    ax_a.set_title("MD-similarity by sampler family (graphene 5x5 pilot)")
    ax_a.grid(alpha=0.3)
    fig_a.tight_layout()
    fig_a.savefig(OUTPUT_ROOT / "fig_a_amp_vs_corr.png", dpi=200)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    for r in results:
        if r["D_corr"] is None:
            continue
        ax_b.scatter(r["D_corr"], r["H_MAE_mean"] * 1000, s=80, color=FAMILY_COLORS.get(r["family"], "#000"), label=r["family"])
        ax_b.annotate(r["family"], (r["D_corr"], r["H_MAE_mean"] * 1000), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax_b.set_xlabel("D_corr (distance to MD's shell correlation)")
    ax_b.set_ylabel("H-MAE (meV, mean over pilot rows)")
    ax_b.set_title("Does MD-similarity predict Hamiltonian accuracy?")
    ax_b.grid(alpha=0.3)
    fig_b.tight_layout()
    fig_b.savefig(OUTPUT_ROOT / "fig_b_corr_vs_hmae.png", dpi=200)
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(6, 5))
    for r in results:
        if r["D_corr"] is None or r["rel_Frob_mean"] is None:
            continue
        ax_c.scatter(r["D_corr"], r["rel_Frob_mean"] * 100, s=80, color=FAMILY_COLORS.get(r["family"], "#000"), label=r["family"])
        ax_c.annotate(r["family"], (r["D_corr"], r["rel_Frob_mean"] * 100), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax_c.set_xlabel("D_corr (distance to MD's shell correlation)")
    ax_c.set_ylabel("Relative Frobenius error (%, mean over pilot rows)")
    ax_c.set_title("Does MD-similarity predict relative Frobenius error?")
    ax_c.grid(alpha=0.3)
    fig_c.tight_layout()
    fig_c.savefig(OUTPUT_ROOT / "fig_c_corr_vs_relfrob.png", dpi=200)
    plt.close(fig_c)

    print(json.dumps({"done": True, "csv": str(csv_path),
                       "figures": [str(OUTPUT_ROOT / "fig_a_amp_vs_corr.png"), str(OUTPUT_ROOT / "fig_b_corr_vs_hmae.png"),
                                   str(OUTPUT_ROOT / "fig_c_corr_vs_relfrob.png")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
