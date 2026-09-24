#!/usr/bin/env python3
"""Exploratory 6x6 train/validation/test label-budget study (no new SIESTA)."""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import numpy as np

import dataset_design_curves_v1 as c


ROOT = c.OUT / "label_budget_6x6"
RUNS = ROOT / "runs"
REPORT = c.REPO_ROOT / "docs/dataset_design_label_budget_6x6.md"
TRAIN_SIZES = (4, 8, 16, 64)
VAL_SIZES = (8, 24, 48)
TEST_SIZES = (4, 8, 16, 24, 32, 48)
METRICS = ("H_MAE_meV", "band_rmse_meV", "dos_rel_L1")
SELECTOR_SEED = 20260924
SUBSAMPLE_SEED = 20260925
REPEATS = 10_000


def validation_sets(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Balanced metadata-only V8 subset of V24 subset of the frozen V48."""

    assert len(rows) == 48 and len({row["sample_id"] for row in rows}) == 48
    families = sorted({row["family"] for row in rows})
    assert len(families) == 3
    family_index = {value: index for index, value in enumerate(families)}

    def key(row: dict[str, Any]) -> tuple[int, int, int, str]:
        return (family_index[row["family"]], c.DIMS.index(row["dim"]), int(row["stratum"]), row["sample_id"])

    ordered = sorted(rows, key=key)
    v24 = [row for row in ordered if sum(key(row)[:3]) % 2 == SELECTOR_SEED % 2]
    assert len(v24) == 24
    v24 = [v24[index] for index in np.random.default_rng(SELECTOR_SEED).permutation(len(v24))]
    v8 = None
    # ponytail: 24 choose 8 is small; a deterministic exhaustive search is clearer than an optimizer.
    for candidate in combinations(v24, 8):
        dims = Counter(row["dim"] for row in candidate)
        strata = Counter(int(row["stratum"]) for row in candidate)
        family_counts = Counter(row["family"] for row in candidate)
        if (set(dims.values()) == {2} and len(dims) == 4
                and set(strata.values()) == {2} and len(strata) == 4
                and sorted(family_counts.values()) == [2, 3, 3]):
            v8 = list(candidate)
            break
    assert v8 is not None
    return {8: v8, 24: v24, 48: ordered}


def dimension_orders(test_rows: list[dict[str, Any]], repeats: int = REPEATS,
                     seed: int = SUBSAMPLE_SEED) -> dict[str, np.ndarray]:
    """One paired random order per dimensionality and repeat."""

    rng, output = np.random.default_rng(seed), {}
    for dim in c.DIMS:
        ids = np.array(sorted(row["sample_id"] for row in test_rows if row["dim"] == dim), dtype=object)
        assert len(ids) == 12
        output[dim] = np.take_along_axis(
            np.broadcast_to(ids, (repeats, len(ids))),
            np.argsort(rng.random((repeats, len(ids))), axis=1), axis=1,
        )
    return output


def subsample_means(values: dict[str, float], orders: dict[str, np.ndarray], n_test: int) -> np.ndarray:
    """Equal-dimension aggregate; Test48 has equally sized dimensional strata."""

    per_dim = n_test // len(c.DIMS)
    assert n_test % len(c.DIMS) == 0 and 1 <= per_dim <= 12
    strata = [np.vectorize(values.__getitem__, otypes=[float])(orders[dim][:, :per_dim]).mean(axis=1)
              for dim in c.DIMS]
    return np.stack(strata).mean(axis=0)


def choose_minimum(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [row for row in rows if row["pass"]]
    return min(passed, key=lambda row: (row["N_total"], row["H_MAE_conservative_meV"],
                                        row["N_train"], row["N_val"], row["N_test"])) if passed else None


def disjoint_hashes(*groups: list[dict[str, Any]]) -> bool:
    sets = [{row["hash"] for row in group} for group in groups]
    return all(not sets[left] & sets[right] for left in range(len(sets)) for right in range(left + 1, len(sets)))


def _mean_metrics(path: Path) -> dict[str, float]:
    rows = c.read_csv(path)
    assert len(rows) == 48
    return {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}


def _existing_model(n_train: int) -> dict[str, Any]:
    winner = c.read_json(c.OUT / "coverage_6x6_winner.json")["recipe"]["recipe_id"]
    matches = [row for row in c.load_results("6x6")
               if row["recipe_id"] == winner and int(row["N"]) == n_train
               and int(row["training_seed"]) == 0
               and row["stage"] in {"coverage_6x6", "coverage_curve_6x6"}]
    assert len(matches) == 1, (n_train, [row["job_id"] for row in matches])
    path = c.OUT / "runs/6x6" / matches[0]["job_id"] / "final_test/per_structure_metrics.csv"
    if not path.exists():
        raise RuntimeError(f"missing current-sweep Test48 evaluation: {path}")
    return matches[0] | {"final_test_csv": str(path)}


def _composition(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    output = []
    for field in ("family", "dim", "stratum"):
        output += [{"N_val": size, "field": field, "level": level, "count": count}
                   for level, count in sorted(Counter(str(row[field]) for row in rows).items())]
    return output


def prepare_manifest() -> dict[str, Any]:
    path = ROOT / "manifest.json"
    if path.exists():
        return c.read_json(path)

    winner = c.read_json(c.OUT / "coverage_6x6_winner.json")
    train_rows = winner["samples"]
    val_rows = c.read_json(c.OUT / "common_dev_6x6_v1.json")["samples"]
    test_rows = c.read_json(c.OUT / "final_test_manifest.json")["systems"]["6x6"]["samples"]
    sets = validation_sets(val_rows)
    assert disjoint_hashes(train_rows, val_rows, test_rows)

    existing = {n: _existing_model(n) for n in (4, 8, 16, 32, 64)}
    full = {n: _mean_metrics(Path(existing[n]["final_test_csv"])) for n in (16, 32, 64)}
    guards = {"H_MAE_meV": 1.05, "band_rmse_meV": 1.10, "dos_rel_L1": 1.10}
    ratios = {metric: full[16][metric] / full[64][metric] for metric in METRICS}
    include_32 = any(ratios[metric] > guards[metric] for metric in METRICS)
    train_sizes = list(TRAIN_SIZES[:-1]) + ([32] if include_32 else []) + [64]

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "exploratory minimum for synthetic graphene 6x6; no new SIESTA labels",
        "recipe": winner["recipe"], "training_seed": 0,
        "training": {"batch_size": 16, "optimizer_updates": 8000, "validation_checks": 2000,
                     "loss": "graph2mat.metrics.elementwise_mse", "scheduler": "cosine"},
        "train_sizes": train_sizes, "validation_sizes": list(VAL_SIZES), "test_sizes": list(TEST_SIZES),
        "selector_seed": SELECTOR_SEED,
        "validation_sets": {str(size): {"sample_ids": [row["sample_id"] for row in selected],
                                                "geometry_hashes": [row["hash"] for row in selected],
                                                "ids_hash": c.ids_hash([row["sample_id"] for row in selected])}
                            for size, selected in sets.items()},
        "train_prefixes": {str(size): {"sample_ids": [row["sample_id"] for row in train_rows[:size]],
                                            "geometry_hashes": [row["hash"] for row in train_rows[:size]],
                                            "ids_hash": c.ids_hash([row["sample_id"] for row in train_rows[:size]])}
                           for size in train_sizes},
        "test": {"sample_ids": [row["sample_id"] for row in test_rows],
                 "geometry_hashes": [row["hash"] for row in test_rows],
                 "ids_hash": c.ids_hash([row["sample_id"] for row in test_rows]),
                 "subsampling_seed": SUBSAMPLE_SEED, "repetitions": REPEATS,
                 "method": "paired nested random permutations within each of four dimensions"},
        "disjoint_geometry_hashes": True,
        "conditional_N32": {"rule": "include if N16 exceeds N64 by H 5%, band 10%, or DOS 10% on Test48",
                            "full_test_metrics": full, "N16_over_N64": ratios, "included": include_32},
        "reference": {"N_train": 64, "N_val": 48, "N_test": 48, "training_seed": 0},
        "acceptance": {"H_MAE_ratio_max": 1.05, "band_rmse_ratio_max": 1.10,
                       "dos_rel_L1_ratio_max": 1.10, "H_subsampling_q90_abs_max": 0.10,
                       "ranking": "no systematic inversion (>50%) when full-test difference is at least 5% of reference"},
        "existing_V48_models": {str(n): existing[n]["job_id"] for n in train_sizes},
    }
    c.write_json(path, manifest)
    c.write_csv(ROOT / "validation_composition.csv",
                [row for size, selected in sets.items() for row in _composition(selected, size)])
    return manifest


def _rows_by_ids(rows: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    indexed = {row["sample_id"]: row for row in rows}
    return [indexed[sample_id] for sample_id in ids]


def _new_result_path(n_train: int, n_val: int) -> Path:
    return RUNS / f"N{n_train}__V{n_val}__seed0" / "result.json"


def train_model(n_train: int, n_val: int, manifest: dict[str, Any], accelerator: str,
                concurrency: int) -> dict[str, Any]:
    result_path = _new_result_path(n_train, n_val)
    if result_path.exists():
        return c.read_json(result_path)
    run_dir, done_path = result_path.parent, result_path.parent / "train_done.json"
    train_all = c.read_json(c.OUT / "coverage_6x6_winner.json")["samples"]
    val_all = c.read_json(c.OUT / "common_dev_6x6_v1.json")["samples"]
    train_rows = train_all[:n_train]
    val_rows = _rows_by_ids(val_all, manifest["validation_sets"][str(n_val)]["sample_ids"])
    train, val = [c.sample_from_dict(row) for row in train_rows], [c.sample_from_dict(row) for row in val_rows]
    if done_path.exists():
        done = c.read_json(done_path)
        checkpoint, seconds, peak = Path(done["checkpoint"]), float(done["seconds"]), int(done["peak"])
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        job_id = run_dir.name
        config = c.s4.build_graph2mat_config(run_dir, train, val, run_name=job_id, accelerator=accelerator,
                                             training_seed=0, **c.TRAIN_KW)
        c.apply_config_patch(config, c.fixed_update_elemmse_patch("6x6", n_train))
        checkpoint, seconds, peak = c.run_training(config, run_dir)
        c.write_json(done_path, {"checkpoint": str(checkpoint), "seconds": seconds, "peak": peak})
    epochs = c.tensorboard_epochs(run_dir, checkpoint)
    val_metrics = c.summarize(c.evaluate(checkpoint, val, run_dir / "val_eval", accelerator))
    step_match = re.search(r"step=?(\d+)", checkpoint.name)
    result = {
        "system": "6x6", "stage": "label_budget_6x6", "job_id": run_dir.name,
        "recipe_id": manifest["recipe"]["recipe_id"], "N_train": n_train, "N_val": n_val,
        "training_seed": 0, "train_ids_hash": manifest["train_prefixes"][str(n_train)]["ids_hash"],
        "val_ids_hash": manifest["validation_sets"][str(n_val)]["ids_hash"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": c.sha256_file(checkpoint),
        "checkpoint_policy": "best_val_loss (ModelCheckpoint save_top_k=1)",
        "best_update": int(step_match.group(1)) if step_match else None, **epochs,
        "train_seconds": seconds, "gpu_h": seconds / 3600, "concurrent_jobs": concurrency,
        "peak_gpu_mib": peak,
        "siesta_cpu_h_train": sum(float(row["siesta_cpu_s"]) for row in train_rows) / 3600,
        "val_metrics": val_metrics,
    }
    c.write_json(result_path, result)
    return result


def evaluate_model(model: dict[str, Any], test_rows: list[dict[str, Any]], accelerator: str) -> Path:
    out_dir = Path(model["result_path"]).parent / "final_test"
    path = out_dir / "per_structure_metrics.csv"
    if path.exists():
        return path
    checkpoint = Path(model["checkpoint"])
    assert c.sha256_file(checkpoint) == model["checkpoint_sha256"]
    samples = [c.sample_from_dict(row) for row in test_rows]
    predicted_root = c.predict(checkpoint, samples, out_dir, accelerator)
    metadata = {row["sample_id"]: row for row in test_rows}
    rows = []
    for sample in samples:
        row = c.structure_metrics(sample, predicted_root)
        if row is None:
            raise RuntimeError(f"missing prediction: {sample.sample_id}")
        row |= c.band_dos_metrics(sample, predicted_root / sample.sample_id / "ML_prediction.HSX", "6x6")
        meta = metadata[sample.sample_id]
        row |= {"dim": meta["dim"], "amplitude_ang": meta["amplitude_ang"], "replica": meta["replica"]}
        rows.append(row)
    c.write_csv(path, rows)
    shutil.rmtree(predicted_root, ignore_errors=True)
    return path


def collect_models(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for n_train in manifest["train_sizes"]:
        existing = _existing_model(int(n_train))
        for n_val in VAL_SIZES:
            if n_val == 48:
                output.append({"model_id": f"N{n_train}__V48__seed0", "N_train": int(n_train), "N_val": 48,
                               "source": "reused current sweep", "checkpoint": existing["checkpoint"],
                               "checkpoint_sha256": existing["checkpoint_sha256"],
                               "best_epoch": existing.get("best_epoch"),
                               "best_update": _checkpoint_step(existing["checkpoint"]),
                               "train_seconds": existing["train_seconds"],
                               "siesta_cpu_h_train": existing["siesta_cpu_h_train"],
                               "val_H_MAE_meV": existing["val_metrics"]["H_MAE_meV"],
                               "result_path": str(c.OUT / "runs/6x6" / existing["job_id"] / "result.json"),
                               "final_test_csv": existing["final_test_csv"]})
            else:
                result_path = _new_result_path(int(n_train), n_val)
                result = c.read_json(result_path)
                output.append({"model_id": result["job_id"], "N_train": int(n_train), "N_val": n_val,
                               "source": "new label-budget run", "checkpoint": result["checkpoint"],
                               "checkpoint_sha256": result["checkpoint_sha256"],
                               "best_epoch": result.get("best_epoch"), "best_update": result.get("best_update"),
                               "train_seconds": result["train_seconds"],
                               "siesta_cpu_h_train": result["siesta_cpu_h_train"],
                               "val_H_MAE_meV": result["val_metrics"]["H_MAE_meV"],
                               "result_path": str(result_path),
                               "final_test_csv": str(result_path.parent / "final_test/per_structure_metrics.csv")})
    return output


def _checkpoint_step(checkpoint: str) -> int | None:
    match = re.search(r"step=?(\d+)", Path(checkpoint).name)
    return int(match.group(1)) if match else None


def analyze(manifest: dict[str, Any], models: list[dict[str, Any]]) -> None:
    test_rows = c.read_json(c.OUT / "final_test_manifest.json")["systems"]["6x6"]["samples"]
    orders = dimension_orders(test_rows)
    full_rows, per_structure, stats_rows, draws = [], [], [], {}
    for model in models:
        rows = c.read_csv(Path(model["final_test_csv"]))
        assert len(rows) == 48 and {row["sample_id"] for row in rows} == set(manifest["test"]["sample_ids"])
        full = {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}
        full_rows.append(model | full)
        metadata = {row["sample_id"]: row for row in test_rows}
        per_structure += [{"model_id": model["model_id"], "N_train": model["N_train"], "N_val": model["N_val"],
                           "sample_id": row["sample_id"], "dim": metadata[row["sample_id"]]["dim"],
                           "amplitude_ang": metadata[row["sample_id"]]["amplitude_ang"],
                           "replica": metadata[row["sample_id"]]["replica"],
                           **{metric: float(row[metric]) for metric in METRICS}} for row in rows]
        for metric in METRICS:
            values = {row["sample_id"]: float(row[metric]) for row in rows}
            for n_test in TEST_SIZES:
                sampled = subsample_means(values, orders, n_test)
                delta = (sampled - full[metric]) / full[metric]
                draws[(model["model_id"], n_test, metric)] = sampled
                stats_rows.append({"model_id": model["model_id"], "N_train": model["N_train"],
                                   "N_val": model["N_val"], "N_test": n_test, "metric": metric,
                                   "full_test_value": full[metric], "delta_median": float(np.median(delta)),
                                   "delta_p05": float(np.quantile(delta, .05)),
                                   "delta_p95": float(np.quantile(delta, .95)),
                                   "q90_abs_delta": float(np.quantile(np.abs(delta), .90)),
                                   "P_abs_delta_le_10pct": float(np.mean(np.abs(delta) <= .10)),
                                   "interval_type": "empirical subsampling interval"})

    reference = next(row for row in full_rows if row["N_train"] == 64 and row["N_val"] == 48)
    stats = {(row["model_id"], row["N_test"], row["metric"]): row for row in stats_rows}
    ranking = []
    for left, right in combinations(full_rows, 2):
        full_diff = left["H_MAE_meV"] - right["H_MAE_meV"]
        relevant = abs(full_diff) >= .05 * reference["H_MAE_meV"]
        for n_test in TEST_SIZES:
            sampled_diff = (draws[(left["model_id"], n_test, "H_MAE_meV")]
                            - draws[(right["model_id"], n_test, "H_MAE_meV")])
            preserved = float(np.mean(np.sign(sampled_diff) == np.sign(full_diff))) if full_diff else 1.0
            ranking.append({"model_A": left["model_id"], "model_B": right["model_id"], "N_test": n_test,
                            "full_H_difference_meV": full_diff, "difference_relevant": relevant,
                            "rank_preservation_probability": preserved,
                            "rank_inversion_frequency": 1 - preserved})
    cartesian = []
    for model in full_rows:
        quality = {"H_guard": model["H_MAE_meV"] <= 1.05 * reference["H_MAE_meV"],
                   "band_guard": model["band_rmse_meV"] <= 1.10 * reference["band_rmse_meV"],
                   "DOS_guard": model["dos_rel_L1"] <= 1.10 * reference["dos_rel_L1"]}
        for n_test in TEST_SIZES:
            h_stats = stats[(model["model_id"], n_test, "H_MAE_meV")]
            reliability = h_stats["q90_abs_delta"] <= .10
            relevant_pairs = [row for row in ranking if row["N_test"] == n_test and row["difference_relevant"]
                              and model["model_id"] in (row["model_A"], row["model_B"])]
            worst_rank_preservation = min((row["rank_preservation_probability"] for row in relevant_pairs), default=1.0)
            ranking_guard = worst_rank_preservation >= .50
            full_diff = model["H_MAE_meV"] - reference["H_MAE_meV"]
            relevant = abs(full_diff) >= .05 * reference["H_MAE_meV"]
            if model["model_id"] == reference["model_id"] or not relevant:
                rank_probability = None
            else:
                sampled_diff = (draws[(model["model_id"], n_test, "H_MAE_meV")]
                                - draws[(reference["model_id"], n_test, "H_MAE_meV")])
                rank_probability = float(np.mean(np.sign(sampled_diff) == np.sign(full_diff)))
            passed = all(quality.values()) and reliability and ranking_guard
            reasons = [name for name, ok in (quality | {"test_reliability": reliability,
                                                        "ranking_guard": ranking_guard}).items() if not ok]
            cartesian.append({"model_id": model["model_id"], "N_train": model["N_train"],
                              "N_val": model["N_val"], "N_test": n_test,
                              "N_total": model["N_train"] + model["N_val"] + n_test,
                              **{metric: model[metric] for metric in METRICS},
                              "H_ratio_to_reference": model["H_MAE_meV"] / reference["H_MAE_meV"],
                              "band_ratio_to_reference": model["band_rmse_meV"] / reference["band_rmse_meV"],
                              "DOS_ratio_to_reference": model["dos_rel_L1"] / reference["dos_rel_L1"],
                              "H_q90_abs_delta": h_stats["q90_abs_delta"],
                              "H_P_abs_delta_le_10pct": h_stats["P_abs_delta_le_10pct"],
                              "H_MAE_conservative_meV": model["H_MAE_meV"] * (1 + h_stats["q90_abs_delta"]),
                              **quality, "test_reliability": reliability,
                              "worst_rank_preservation": worst_rank_preservation,
                              "ranking_guard": ranking_guard,
                              "rank_preservation_vs_reference": rank_probability,
                              "rank_difference_relevant": relevant, "pass": passed,
                              "failure_reasons": ";".join(reasons) if reasons else "none"})

    minimum = choose_minimum(cartesian)
    c.write_csv(ROOT / "model_results.csv", full_rows)
    c.write_csv(ROOT / "per_structure_results.csv", per_structure)
    c.write_csv(ROOT / "test_subsampling.csv", stats_rows)
    c.write_csv(ROOT / "ranking_stability.csv", ranking)
    c.write_csv(ROOT / "cartesian_results.csv", cartesian)
    c.write_json(ROOT / "minimum.json", {
        "status": "minimum found" if minimum else "minimum not reached",
        "minimum": minimum, "reference": {key: reference[key] for key in ("model_id", "N_train", "N_val", *METRICS)},
        "criteria": manifest["acceptance"],
        "interpretation": "exploratory minimum for synthetic graphene 6x6; future independent 6x6 MD is confirmatory",
    })
    write_report(manifest, full_rows, cartesian, minimum, reference)


def write_report(manifest: dict[str, Any], models: list[dict[str, Any]], cartesian: list[dict[str, Any]],
                 minimum: dict[str, Any] | None, reference: dict[str, Any]) -> None:
    best = min(models, key=lambda row: row["H_MAE_meV"])
    selected = (f"Ntrain={minimum['N_train']}, Nval={minimum['N_val']}, Ntest={minimum['N_test']} "
                f"(Ntotal={minimum['N_total']})" if minimum else "mínimo no alcanzado")
    table = "\n".join(
        ["| Ntrain | Nval | H-MAE (meV) | Band RMSE (meV) | DOS L1 | Mejor update |",
         "|---:|---:|---:|---:|---:|---:|"]
        + [f"| {row['N_train']} | {row['N_val']} | {row['H_MAE_meV']:.3f} | "
           f"{row['band_rmse_meV']:.2f} | {row['dos_rel_L1']:.4f} | {row.get('best_update') or '—'} |"
           for row in sorted(models, key=lambda item: (item["N_train"], item["N_val"]))]
    )
    report = f"""# Presupuesto de etiquetas 6×6 para Graph2Mat

Generado el {time.strftime('%Y-%m-%d %H:%M %Z')}. Estudio exploratorio; no se generaron etiquetas SIESTA nuevas.

## Resultado ejecutivo

- Combinación mínima según las reglas congeladas: **{selected}**.
- Cumplen el criterio estricto de equivalencia **{sum(row['pass'] for row in cartesian)}/{len(cartesian)}** celdas del producto cartesiano.
- Referencia N64/V48/Test48: H-MAE **{reference['H_MAE_meV']:.3f} meV**, bandas **{reference['band_rmse_meV']:.2f} meV**, DOS L1 **{reference['dos_rel_L1']:.4f}**.
- Menor H-MAE observado: N{best['N_train']}/V{best['N_val']}, **{best['H_MAE_meV']:.3f} meV**.
- Ntrain=32 fue **{'incluido' if manifest['conditional_N32']['included'] else 'omitido'}** por la regla previa basada en Test48.

## Método

Se cruzaron Ntrain={manifest['train_sizes']}, Nval={manifest['validation_sizes']} y Ntest={manifest['test_sizes']} con seed de entrenamiento 0. V8⊂V24⊂V48 se eligió solo por familia, dimensionalidad y amplitud. Todos los modelos usan el mismo pool Sobol 3D R=0.12, `elementwise_mse`, batch 16, 8000 actualizaciones, 2000 validaciones y scheduler cosine.

Cada modelo se evaluó una vez en Test48. Los tamaños de test se obtuvieron mediante 10.000 permutaciones emparejadas y anidadas, equilibradas por dimensionalidad. Los percentiles descritos son **intervalos empíricos de submuestreo**, no intervalos de confianza de generalización.

Criterio operativo de equivalencia: H≤1.05×referencia, bandas≤1.10×, DOS≤1.10×, q90(|Δ H|)≤10 % y ausencia de inversión sistemática (más del 50 %) entre modelos cuya diferencia completa supera el 5 %. Quedar fuera de estos márgenes no invalida un resultado ni significa que el modelo sea malo; solo impide considerarlo equivalente a N64/V48 para seleccionar el presupuesto mínimo. Las diferencias menores no se usan para forzar un ranking.

## Test48 por modelo

{table}

## Interpretación y límites

El mínimo reduce el número de etiquetas SIESTA necesarias para train+val+test dentro de este universo sintético 6×6. El test se ha consultado históricamente y la selección usa una sola seed: el resultado debe llamarse **mínimo exploratorio para grafeno 6×6 sintético**. Una trayectoria MD 6×6 independiente y una segunda seed del mínimo serían la confirmación posterior; no forman parte de este barrido.

## Artefactos

- `Comparison/results/dataset_design_curves_v1/label_budget_6x6/manifest.json`
- `model_results.csv`, `test_subsampling.csv`, `cartesian_results.csv`, `minimum.json`
- UI: Dataset Design → Resultados científicos consolidados.
"""
    REPORT.write_text(report, encoding="utf-8")


def wait_for_pid(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        print(f"waiting for current 6x6 sweep PID {pid}", flush=True)
        time.sleep(30)


def main(wait_pid: int | None, parallel: int) -> None:
    if wait_pid:
        wait_for_pid(wait_pid)
    print(f"starting label-budget sweep; training parallel={parallel}", flush=True)
    manifest = prepare_manifest()
    accelerator = c.s4.torch_backend_preflight()["effective_backend"]
    jobs = [(int(n_train), n_val) for n_train in manifest["train_sizes"] for n_val in (8, 24)
            if not _new_result_path(int(n_train), n_val).exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(train_model, n_train, n_val, manifest, accelerator, parallel): (n_train, n_val)
                   for n_train, n_val in jobs}
        for future in concurrent.futures.as_completed(futures):
            n_train, n_val = futures[future]
            result = future.result()
            print(json.dumps({"trained": [n_train, n_val], "H_val_meV": result["val_metrics"]["H_MAE_meV"],
                              "best_update": result.get("best_update")}), flush=True)

    models = collect_models(manifest)
    test_rows = c.read_json(c.OUT / "final_test_manifest.json")["systems"]["6x6"]["samples"]
    pending = [model for model in models if not Path(model["final_test_csv"]).exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for path in pool.map(lambda model: evaluate_model(model, test_rows, accelerator), pending):
            print(f"evaluated {path.parent.parent.name}", flush=True)
    analyze(manifest, models)
    print("LABEL BUDGET COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    main(args.wait_for_pid, args.parallel)
