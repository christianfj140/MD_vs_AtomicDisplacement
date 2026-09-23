#!/usr/bin/env python3
"""Complete the 6x6 coverage comparison and the winner's learning curve."""
import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataset_design_curves_v1 as c


METRICS = ("H_MAE_meV", "rel_Frob", "band_rmse_meV", "dos_rel_L1")


def prepare_pool() -> None:
    path = c.OUT / "coverage_pool_6x6_r012.json"
    parent, rows = c._finalist("6x6", "efficient")
    assert parent["dim"] == "3D" and parent["R"] == 0.08 and len(rows) == 64
    geometry, base = c.sampler.load_graphene_6x6(), c.base_positions("6x6")
    assert np.allclose(geometry.positions_ang, base, atol=1e-8)
    active = c.sampler.select_active_atoms(geometry, 72)
    forbidden = set(c.read_json(c.OUT / "training_hashes_6x6.json"))
    forbidden |= {s["hash"] for s in c.read_json(c.OUT / "common_dev_6x6_v1.json")["samples"]}
    forbidden |= {s["hash"] for s in c.read_json(c.OUT / "final_test_manifest.json")["systems"]["6x6"]["samples"]}
    entries, expected = [], []
    for i, row in enumerate(rows):
        positions = np.asarray(c.s5._positions_from_run_fdf(Path(row["run_fdf"])))
        vectors = 1.5 * (positions - base)
        expected.append(base + vectors)
        assert c.s5.geometry_hash(expected[-1]) not in forbidden
        displacements = {j: tuple(v) for j, v in enumerate(vectors)}
        metadata = c.sampler.build_metadata(geometry, active, displacements,
                                            family="sobol_sparse", dim="3D", amplitude_ang=0.12)
        entries.append((f"coverage_6x6_3D_R012__pt{i:04d}",
                        c.sampler.Configuration("sobol_sparse", displacements, metadata)))
    if not path.exists():
        refs = c.label_structures(entries, c.OUT / "coverage_6x6_r012", c.sampler.GRAPHENE_6X6_FDF, 2)
        samples = [c.labelled_sample("6x6", sid, refs, {"family": "sobol_sparse", "dim": "3D",
                   "k": 72, "amplitude_ang": 0.12}) for sid, _ in entries]
        hashes = [row["hash"] for row in samples]
        assert len(set(hashes)) == 64 and not set(hashes) & forbidden
        c.write_json(path, {"recipe_id": "sobol_sparse__3D__R0.12__d64__scaled",
                            "family": "sobol_sparse", "dim": "3D", "k": 72, "R": 0.12,
                            "sampler_seed": parent["sampler_seed"], "pool_hash": c.ids_hash(hashes),
                            "parent_recipe_id": parent["recipe_id"], "parent_pool_hash": parent["pool_hash"],
                            "displacement_scale": 1.5, "samples": samples})
    pool = c.read_json(path)
    assert pool["parent_pool_hash"] == parent["pool_hash"] and len(pool["samples"]) == 64
    for row, target in zip(pool["samples"], expected):
        assert np.allclose(c.s5._positions_from_run_fdf(Path(row["run_fdf"])), target, atol=1e-8)
        assert Path(row["reference_matrix"]).exists()
        assert (Path(row["reference_dir"]) / "0_NORMAL_EXIT").exists()
        assert row["hash"] not in forbidden and row["siesta_cpu_s"] > 0
    print("6x6 pool verified: 64 paired 3D labels at R=0.12; train/dev/test disjoint", flush=True)


def select_winner() -> None:
    models = [r for r in c.load_results("6x6") if r["stage"] == "coverage_6x6"]
    assert len(models) == 15
    grouped = {}
    for recipe in sorted({r["recipe_id"] for r in models}):
        rows = [r for r in models if r["recipe_id"] == recipe]
        assert sorted(r["training_seed"] for r in rows) == list(range(5))
        grouped[recipe] = {
            "dev_H_MAE_meV_mean": float(np.mean([r["val_metrics"]["H_MAE_meV"] for r in rows])),
            "dev_H_MAE_meV_sd": float(np.std([r["val_metrics"]["H_MAE_meV"] for r in rows], ddof=1)),
        }
    winner_id = min(grouped, key=lambda recipe: grouped[recipe]["dev_H_MAE_meV_mean"])
    pools = c.read_json(c.OUT / "pools_6x6.json")
    expanded = c.read_json(c.OUT / "coverage_pool_6x6_r012.json")
    selected = expanded if winner_id == expanded["recipe_id"] else pools[winner_id]
    recipe = {key: selected[key] for key in ("recipe_id", "family", "dim", "k", "R", "sampler_seed", "pool_hash")}
    c.write_json(c.OUT / "coverage_6x6_winner.json", {
        "selection_rule": "lowest five-seed mean H-MAE on common_dev_6x6_v1; final test not consulted",
        "candidate_dev_metrics": grouped, "recipe": recipe, "samples": selected["samples"],
    })
    print({"6x6_coverage_winner": winner_id, **grouped[winner_id]}, flush=True)


def _model_cube(models: list[dict], samples: list[dict]) -> tuple[dict, list[dict]]:
    ids = [sample["sample_id"] for sample in samples]
    cubes, output = {}, []
    for model in models:
        path = c.OUT / "runs/6x6" / model["job_id"] / "final_test/per_structure_metrics.csv"
        indexed = {row["sample_id"]: row for row in c.read_csv(path)}
        assert len(indexed) == len(ids) and set(indexed) == set(ids)
        values = np.array([[float(indexed[sid][metric]) for metric in METRICS] for sid in ids])
        assert np.isfinite(values).all()
        cubes[(model["recipe_id"], model["N"], model["training_seed"])] = values
        output.append({"recipe": model["recipe_id"], "N": model["N"], "seed": model["training_seed"],
                       "n_structures": len(ids), "dev_H_MAE_meV": model["val_metrics"]["H_MAE_meV"],
                       "siesta_cpu_h_train": model["siesta_cpu_h_train"],
                       "train_wall_minutes": model["train_seconds"] / 60,
                       **dict(zip(METRICS, values.mean(axis=0)))})
    return cubes, output


def _bootstrap_delta(left: np.ndarray, right: np.ndarray, seed: int) -> list[dict]:
    delta, rng = left - right, np.random.default_rng(seed)
    boot = np.empty((10_000, len(METRICS)))
    for i in range(len(boot)):
        seeds = rng.integers(0, 5, 5)
        structures = rng.integers(0, delta.shape[1], delta.shape[1])
        boot[i] = delta[seeds][:, structures].mean(axis=(0, 1))
    return [{"metric": metric, "mean_delta": float(delta[:, :, j].mean()),
             "ci95_low": float(np.percentile(boot[:, j], 2.5)),
             "ci95_high": float(np.percentile(boot[:, j], 97.5)),
             "better_seeds": int((delta[:, :, j].mean(axis=1) < 0).sum())}
            for j, metric in enumerate(METRICS)]


def summarize() -> None:
    samples = c.read_json(c.OUT / "final_test_manifest.json")["systems"]["6x6"]["samples"]
    coverage = [r for r in c.load_results("6x6") if r["stage"] == "coverage_6x6"]
    winner = c.read_json(c.OUT / "coverage_6x6_winner.json")["recipe"]["recipe_id"]
    curve = [r for r in c.load_results("6x6") if
             (r["stage"] == "coverage_curve_6x6" or
              (r["stage"] == "coverage_6x6" and r["recipe_id"] == winner))]
    assert len(coverage) == 15 and len(curve) == 25

    coverage_cubes, coverage_rows = _model_cube(coverage, samples)
    curve_cubes, curve_rows = _model_cube(curve, samples)
    out = c.OUT / "coverage_6x6_physics"
    c.write_csv(out / "per_seed.csv", coverage_rows)
    summary = []
    recipes = sorted({row["recipe"] for row in coverage_rows})
    for recipe in recipes:
        rows = [row for row in coverage_rows if row["recipe"] == recipe]
        item = {"recipe": recipe, "N": 64, "n_seeds": 5}
        for metric in (*METRICS, "dev_H_MAE_meV", "siesta_cpu_h_train", "train_wall_minutes"):
            values = [row[metric] for row in rows]
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_sd"] = float(np.std(values, ddof=1))
        summary.append(item)
        print(item, flush=True)
    c.write_csv(out / "summary.csv", summary)

    arrays = {recipe: np.stack([coverage_cubes[(recipe, 64, seed)] for seed in range(5)])
              for recipe in recipes}
    comparisons = []
    for index, (left, right) in enumerate(((recipes[2], recipes[1]), (recipes[1], recipes[0]),
                                           (recipes[2], recipes[0]))):
        comparisons += [{"A": left, "B": right, **row}
                        for row in _bootstrap_delta(arrays[left], arrays[right], 20260923 + index)]
    c.write_csv(out / "paired_comparisons.csv", comparisons)

    labels = {recipe: recipe.replace("sobol_sparse__", "").replace("__d64", "").replace("__scaled", "")
              for recipe in recipes}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, j, ylabel, scale in zip(axes, (0, 2, 3), ("H-MAE (meV)", "Band RMSE (meV)", "DOS relative L1 (%)"), (1, 1, 100)):
        means = [arrays[recipe][:, :, j].mean() * scale for recipe in recipes]
        sds = [arrays[recipe][:, :, j].mean(axis=1).std(ddof=1) * scale for recipe in recipes]
        ax.errorbar(range(3), means, yerr=sds, fmt="o", capsize=4)
        ax.set(xticks=range(3), xticklabels=[labels[r] for r in recipes], ylabel=ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("6x6, N=64, 8000 updates; common test (mean ± seed sd, n=5)")
    fig.savefig(out / "dataset_comparison.png", dpi=180)
    fig.savefig(out / "dataset_comparison.pdf")
    plt.close(fig)

    curve_out = c.OUT / "coverage_6x6_curve"
    c.write_csv(curve_out / "per_seed.csv", curve_rows)
    curve_arrays = {n: np.stack([curve_cubes[(winner, n, seed)] for seed in range(5)]) for n in c.NS}
    curve_summary = []
    for n, array in curve_arrays.items():
        rows = [row for row in curve_rows if row["N"] == n]
        item = {"recipe": winner, "N": n, "n_seeds": 5, "n_structures": len(samples)}
        for j, metric in enumerate(METRICS):
            values = array[:, :, j].mean(axis=1)
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_sd"] = float(values.std(ddof=1))
        item["train_wall_minutes_mean"] = float(np.mean([row["train_wall_minutes"] for row in rows]))
        curve_summary.append(item)
    c.write_csv(curve_out / "summary.csv", curve_summary)
    paired = []
    for n in (4, 8, 16, 32):
        paired += [{"N": n, "reference_N": 64, **row}
                   for row in _bootstrap_delta(curve_arrays[n], curve_arrays[64], 20260930 + n)]
    c.write_csv(curve_out / "paired_vs_N64.csv", paired)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, j, ylabel, scale in zip(axes, (0, 2, 3), ("H-MAE (meV)", "Band RMSE (meV)", "DOS relative L1 (%)"), (1, 1, 100)):
        means = [curve_arrays[n][:, :, j].mean() * scale for n in c.NS]
        sds = [curve_arrays[n][:, :, j].mean(axis=1).std(ddof=1) * scale for n in c.NS]
        ax.errorbar(c.NS, means, yerr=sds, fmt="o-", capsize=4)
        ax.set(xscale="log", xticks=c.NS, xticklabels=c.NS, xlabel="Training structures (N)", ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"6x6 {labels.get(winner, winner)}; 8000 updates (mean ± seed sd, n=5)")
    fig.savefig(curve_out / "learning_curve.png", dpi=180)
    fig.savefig(curve_out / "learning_curve.pdf")
    plt.close(fig)


def write_report() -> None:
    """Write the presentation-ready report only after every queued result exists."""

    def table(headers: list[str], rows: list[list[object]]) -> str:
        return "\n".join(("| " + " | ".join(headers) + " |",
                          "| " + " | ".join("---" for _ in headers) + " |",
                          *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)))

    def label(recipe: str) -> str:
        if recipe == "md_w90":
            return "MD"
        family = "Sobol" if recipe.startswith("sobol") else "random"
        dim = "3D" if "__3D__" in recipe else "1D in-plane"
        radius = "0.12" if ("r0.120" in recipe or "R0.12" in recipe) else (
                 "0.08" if ("r0.080" in recipe or "R0.08" in recipe) else "0.05")
        return f"{family} {dim} R={radius} Å"

    def summary_rows(path: Path, domain: str | None = None) -> list[dict[str, str]]:
        rows = c.read_csv(path)
        return [row for row in rows if domain is None or row.get("domain") == domain]

    def metric_row(row: dict[str, str]) -> list[object]:
        recipe = row.get("recipe") or row.get("recipe_id")
        return [label(recipe) if recipe else "Sobol 3D R=0.12 Å", row.get("N", 64), row["n_seeds"],
                f"{float(row['H_MAE_meV_mean']):.2f} ± {float(row['H_MAE_meV_sd']):.2f}",
                f"{100 * float(row['rel_Frob_mean']):.2f} ± {100 * float(row['rel_Frob_sd']):.2f}",
                f"{float(row['band_rmse_meV_mean']):.2f} ± {float(row['band_rmse_meV_sd']):.2f}",
                f"{100 * float(row['dos_rel_L1_mean']):.2f} ± {100 * float(row['dos_rel_L1_sd']):.2f}"]

    root = c.OUT
    w90_budget = summary_rows(root / "w90_budget_physics/summary.csv", "all")
    w90_coverage = summary_rows(root / "w90_coverage_physics/summary.csv", "all")
    w90_curve = summary_rows(root / "w90_coverage_curve/summary.csv", "all")
    g6_coverage = summary_rows(root / "coverage_6x6_physics/summary.csv")
    g6_curve = summary_rows(root / "coverage_6x6_curve/summary.csv")
    assert len(w90_budget) == 4 and len(w90_coverage) == 4
    assert len(w90_curve) == len(g6_curve) == 5 and len(g6_coverage) == 3

    w90_best = min(w90_coverage, key=lambda row: float(row["H_MAE_meV_mean"]))
    g6_best = min(g6_coverage, key=lambda row: float(row["H_MAE_meV_mean"]))

    def n_within_five_percent(rows: list[dict[str, str]]) -> int:
        reference = next(float(row["H_MAE_meV_mean"]) for row in rows if int(row["N"]) == 64)
        return min(int(row["N"]) for row in rows if float(row["H_MAE_meV_mean"]) <= 1.05 * reference)

    w90_n, g6_n = n_within_five_percent(w90_curve), n_within_five_percent(g6_curve)
    original = c.read_csv(root / "final_table.csv")
    original_table = table(
        ["Sistema", "Dataset", "N", "H-MAE (meV)", "Bandas RMSE (meV)", "CPU·h SIESTA"],
        [[row["Sistema"], row["Dataset"], row["N"], row["H-MAE (meV)"],
          row["Bandas RMSE (meV)"] or "—", row["CPU·h SIESTA"]] for row in original],
    )
    budget_table = table(
        ["Métrica", "2000 épocas", "4000 épocas", "Reducción", "Semillas que mejoran"],
        [[row["metric"], f"{float(row['mean_2k']):.3f}", f"{float(row['mean_4k']):.3f}",
          f"{float(row['reduction_pct']):.1f}%", f"{row['improved_seeds']}/5"] for row in w90_budget],
    )
    physics_headers = ["Dataset", "N", "semillas", "H-MAE (meV)", "Frobenius relativo (%)", "Bandas RMSE (meV)", "DOS L1 (%)"]
    w90_coverage_table = table(physics_headers, [metric_row(row) for row in w90_coverage])
    w90_curve_table = table(physics_headers, [metric_row(row | {"recipe": w90_best["recipe"]})
                                               for row in sorted(w90_curve, key=lambda r: int(r["N"]))])
    g6_coverage_table = table(physics_headers, [metric_row(row) for row in g6_coverage])
    g6_curve_table = table(physics_headers, [metric_row(row) for row in sorted(g6_curve, key=lambda r: int(r["N"]))])

    w90_pairs = c.read_csv(root / "w90_coverage_physics/paired_comparisons.csv")
    w90_vs_md = [row for row in w90_pairs if row["A"] == "3D R0.12" and row["B"] == "MD" and row["domain"] == "all"]
    paired_table = table(
        ["Métrica", "Δ 3D R=0.12 − MD", "IC95", "Semillas favorables"],
        [[row["metric"], f"{float(row['mean_A_minus_B']):.3f}",
          f"[{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}]", f"{row['better_seeds']}/5"]
         for row in w90_vs_md],
    )

    report = f"""# Informe final — construcción del dataset para Graph2Mat

Generado automáticamente el {time.strftime('%Y-%m-%d %H:%M %Z')} después de completar las campañas w90 y 6×6.

## 1. Pregunta científica

El objetivo es construir el conjunto de entrenamiento más barato que permita a Graph2Mat reproducir Hamiltonianos y observables electrónicos con precisión próxima a la referencia SIESTA/MD. Se estudia la frontera entre coste de etiquetado, número de estructuras, cobertura geométrica y precisión; no se pretende comparar directamente los errores absolutos de w90 (celda primitiva) y 6×6 (72 átomos).

## 2. Resumen ejecutivo

- En **w90**, el mejor dataset probado es **{label(w90_best['recipe'])}**, con H-MAE **{float(w90_best['H_MAE_meV_mean']):.2f} ± {float(w90_best['H_MAE_meV_sd']):.2f} meV**, Frobenius relativo **{100 * float(w90_best['rel_Frob_mean']):.2f} ± {100 * float(w90_best['rel_Frob_sd']):.2f}%**, bandas **{float(w90_best['band_rmse_meV_mean']):.2f} ± {float(w90_best['band_rmse_meV_sd']):.2f} meV** y DOS L1 **{100 * float(w90_best['dos_rel_L1_mean']):.2f} ± {100 * float(w90_best['dos_rel_L1_sd']):.2f}%**.
- Ese dataset cuesta **{float(w90_best['siesta_cpu_h_train_mean']):.3f} CPU·h** de etiquetas, frente a **0.080 CPU·h** del baseline MD. La comparación emparejada con MD se resume abajo y favorece al dataset diseñado en las tres métricas globales.
- La curva w90 sitúa el primer N dentro del 5% de N=64 en **N={w90_n}** para H-MAE. Deben mirarse también bandas y DOS: una H similar no garantiza el mismo espectro.
- En **6×6**, la comparación a presupuesto común selecciona **{label(g6_best['recipe'])}**. Su curva sitúa el primer N dentro del 5% de N=64 en **N={g6_n}**.
- La intervención de entrenamiento decisiva fue cambiar `block_type_mae` por **`elementwise_mse`**: en 6×6 redujo el H-MAE de aproximadamente 5.3 a 2.0 meV. Aumentar ancho, profundidad o resolución angular no produjo una mejora robusta.

## 3. Protocolo común

- Etiquetas electrónicas: SIESTA; modelo: Graph2Mat/MACE.
- Comparaciones finales: cinco semillas de entrenamiento.
- Pérdida: `elementwise_mse`; scheduler cosine.
- Curvas nuevas: **8000 actualizaciones de gradiente y 2000 validaciones** para cada N, evitando confundir épocas con cantidad de optimización.
- Selección de receta con desarrollo común; evaluación posterior sobre el mismo test congelado.
- Métricas: H-MAE, RMSE de bandas Γ–K–M–Γ en ±2 eV de E_F y diferencia L1 relativa de DOS en ±3 eV.
- Incertidumbre: dispersión entre semillas y bootstrap emparejado por semilla y estructura.

## 4. Campaña principal

{original_table}

Esta campaña estableció la frontera inicial, pero empleaba `block_type_mae`. Sus cifras sirven como historial y diagnóstico, no como estimación final de la mejor receta.

## 5. Diagnósticos de entrenamiento

- **Función de pérdida:** `elementwise_mse` mejoró 6×6 por un factor aproximado de 2.6 y redujo fuertemente la dispersión.
- **Presupuesto w90:** duplicar el horizonte de 2000 a 4000 épocas mejoró las cinco semillas, aunque la ganancia espectral fue menor que la ganancia en H.
- **Scheduler:** cosine redujo la varianza; se mantuvo.
- **Ancho, profundidad y `max_ell`:** no mostraron mejora robusta. El modelo 6×6 ancho llegó a ~30.8 GiB de VRAM.
- **Calidad de etiquetas:** el ruido estimado (~0.01 meV) es varios órdenes inferior al error del modelo; no explica el techo.

{budget_table}

## 6. Cobertura del dataset w90

{w90_coverage_table}

Comparación emparejada del mejor diseño frente a MD:

{paired_table}

La ampliación 3D de R=0.08 a R=0.12 mejora de forma resuelta H y DOS. La mejora media de bandas frente a 3D R=0.08 no queda resuelta por el intervalo de confianza. En los propios frames MD la ventaja en H se convierte en empate: la ventaja global proviene principalmente de cubrir desplazamientos fuera del dominio estrecho de estas trayectorias MD.

## 7. Curva coste–precisión w90

{w90_curve_table}

Esta tabla es la base para elegir N: debe usarse el menor N cuyo error y dispersión sean aceptables en **las tres métricas**, no solo H-MAE.

## 8. Comparación y curva 6×6

{g6_coverage_table}

{g6_curve_table}

No se incluye un baseline MD 6×6 porque el repositorio no contiene trayectorias de esa supercelda. Inventarlo o comparar con MD w90 no sería físicamente válido.

## 9. Qué significan los resultados

1. Una vez estabilizada la optimización, **la cobertura geométrica y la amplitud importan más que añadir configuraciones redundantes**.
2. El coste de SIESTA puede reducirse seleccionando un N próximo al punto de saturación; el ahorro exacto debe leerse junto con bandas y DOS.
3. Cinco semillas son necesarias: varios efectos que parecían prometedores con tres semillas desaparecieron al ampliar la réplica.
4. H-MAE es útil para seleccionar, pero solo correlaciona de forma moderada con bandas/DOS; los observables físicos deben validarse explícitamente.

## 10. Limitaciones

- Los seguimientos son exploratorios y reutilizan un test ya consultado; una publicación requeriría un segundo test independiente.
- Las 410 configuraciones MD disponibles proceden de solo seis trayectorias deterministas; no representan seis réplicas estadísticas independientes.
- El dominio MD llega aproximadamente a 0.025 Å, mientras los datasets diseñados cubren hasta 0.08–0.12 Å: parte de la comparación es también una comparación de dominio.
- w90 y 6×6 tienen dimensionalidad y distribución de bloques distintas; no se deben comparar sus H-MAE absolutos como si fueran el mismo problema.
- No se demuestra equivalencia dinámica con MD: se demuestra precisión electrónica sobre un test compartido y una frontera coste–precisión favorable.

## 11. Guion recomendado para explicarlo

1. **Objetivo:** aproximar la referencia electrónica reduciendo el número de cálculos SIESTA.
2. **Control metodológico:** mismo test, cinco semillas, mismo presupuesto en actualizaciones y tres métricas físicas.
3. **Hallazgo de entrenamiento:** `elementwise_mse` elimina el principal techo en 6×6.
4. **Hallazgo de dataset:** 3D y R=0.12 mejoran w90; más cobertura es más útil que densidad redundante.
5. **Frontera coste–precisión:** mostrar las dos curvas y escoger N según el margen físico aceptable.
6. **Cautelas:** MD limitado, seguimiento exploratorio y necesidad de confirmación independiente.

## 12. Artefactos

- UI: pestaña **Dataset Design → Resultados científicos consolidados**.
- Tablas w90: `w90_budget_physics/`, `w90_coverage_physics/`, `w90_coverage_curve/`.
- Tablas 6×6: `coverage_6x6_physics/`, `coverage_6x6_curve/`.
- Figuras principales: `Comparison/results/dataset_design_curves_v1/figures/`.
- Pre-registro y metodología completa: `docs/dataset_design_curves_v1_preregistration.md`.
"""
    path = c.REPO_ROOT / "docs/dataset_design_followup_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"Report ready: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare_pool()
    if not args.prepare_only:
        rc = c.cmd_train(argparse.Namespace(systems=["6x6"], stage=["coverage_6x6"], parallel=2))
        if rc:
            raise SystemExit(rc)
        select_winner()
        rc = c.cmd_train(argparse.Namespace(systems=["6x6"], stage=["coverage_curve_6x6"], parallel=2))
        if rc:
            raise SystemExit(rc)
        c.cmd_evaluate_final(argparse.Namespace(systems=["6x6"],
                            stages=["coverage_6x6", "coverage_curve_6x6"], parallel=2))
        summarize()
        write_report()
        print("6x6 COMPLETE: coverage comparison and five-seed learning curve ready", flush=True)
