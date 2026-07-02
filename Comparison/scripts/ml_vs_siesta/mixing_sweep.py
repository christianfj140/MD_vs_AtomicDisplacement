"""Orchestrate a small/large mixing sweep across dataset sizes × modes × ratios.

``plan_mixing_sweep`` is a pure planner (no IO, no training) that enumerates the
permutations and total sizes. ``run_mixing_sweep`` materializes one merged
dataset per permutation and drives training through an injectable ``launch_fn``
(the real Graph2Mat/DeepH runner in the backend; a fake in tests). This module
never imports the heavy runner and never trains by itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .dataset_mixing import DEFAULT_RATIOS, make_mixed_dataset_manifest
from .mixed_dataset_materialize import (
    dataset_atom_count,
    materialize_mixed_dataset,
    read_dataset_samples,
)

DEFAULT_MODES = ("add", "replace")
_SMALL_PREFIX = "small__"
_LARGE_PREFIX = "large__"

# A launch_fn takes a runner payload and returns a result dict (ideally with a
# "metrics" mapping of model -> {"h_mae_eV": float}).
LaunchFn = Callable[[dict[str, Any]], dict[str, Any]]


def _ratio_slug(ratio: float) -> str:
    return f"r{ratio:.3f}".replace(".", "p")


def plan_mixing_sweep(
    small_counts: dict[int, int],
    large_counts: dict[int, int],
    *,
    sizes: list[int] | None = None,
    modes: tuple[str, ...] = DEFAULT_MODES,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = 0,
) -> dict[str, Any]:
    """Enumerate (size, mode, ratio) permutations with total/composition sizes.

    Pure function: reuses :func:`make_mixed_dataset_manifest` (same add/replace
    arithmetic as the real selection) with synthetic ids of the available counts.
    Sizes lacking a small or large dataset are skipped with a warning.
    """
    small_counts = {int(k): int(v) for k, v in small_counts.items()}
    large_counts = {int(k): int(v) for k, v in large_counts.items()}
    if sizes is None:
        sizes = sorted(set(small_counts) & set(large_counts))
    else:
        sizes = [int(s) for s in sizes]

    permutations: list[dict[str, Any]] = []
    warnings: list[str] = []
    for size in sizes:
        if size not in small_counts or size not in large_counts:
            missing = "small" if size not in small_counts else "large"
            warnings.append(f"size {size}: missing {missing} dataset; skipped")
            continue
        small_ids = [{"id": f"s{i}"} for i in range(small_counts[size])]
        large_ids = [{"id": f"l{i}"} for i in range(large_counts[size])]
        for mode in modes:
            manifest = make_mixed_dataset_manifest(
                small_ids, large_ids, ratios=ratios, mode=mode, seed=seed
            )
            for part in manifest["partitions"]:
                n_large = int(part["n_large_selected"])
                total = int(part["n_selected"])
                permutations.append(
                    {
                        "size": size,
                        "mode": mode,
                        "ratio": float(part["ratio"]),
                        "n_small_available": small_counts[size],
                        "n_large_available": large_counts[size],
                        "n_small_selected": total - n_large,
                        "n_large_selected": n_large,
                        "total_size": total,
                        "large_fraction": (n_large / total) if total else 0.0,
                    }
                )
    return {
        "schema": "ml_vs_siesta_mixing_sweep_plan_v1",
        "sizes": sizes,
        "modes": list(modes),
        "ratios": list(ratios),
        "seed": seed,
        "n_permutations": len(permutations),
        "permutations": permutations,
        "warnings": warnings,
    }


def discover_dataset_sizes(dataset_roots: list[str | Path]) -> dict[int, dict[str, Any]]:
    """Read snapshot count + atom count for each dataset_root (keyed by count)."""
    out: dict[int, dict[str, Any]] = {}
    for root in dataset_roots:
        root_path = Path(root)
        try:
            samples = read_dataset_samples(root_path)
        except Exception:  # noqa: BLE001 - skip non-dataset dirs
            continue
        count = len(samples)
        out[count] = {
            "root": str(root_path),
            "n_snapshots": count,
            "n_atoms": dataset_atom_count(root_path),
        }
    return out


def plan_mixing_sweep_from_roots(
    small_by_size: dict[int, str | Path],
    large_by_size: dict[int, str | Path],
    *,
    sizes: list[int] | None = None,
    modes: tuple[str, ...] = DEFAULT_MODES,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = 0,
) -> dict[str, Any]:
    """Read counts from dataset roots, then plan."""
    small_counts = {int(k): len(read_dataset_samples(v)) for k, v in small_by_size.items()}
    large_counts = {int(k): len(read_dataset_samples(v)) for k, v in large_by_size.items()}
    return plan_mixing_sweep(
        small_counts, large_counts, sizes=sizes, modes=modes, ratios=ratios, seed=seed
    )


def _build_runner_payload(
    dataset_root: Path,
    result_dir: Path,
    models: tuple[str, ...],
    *,
    epochs: int | None,
    system_label: str,
) -> dict[str, Any]:
    """Minimal 'reuse_validated' payload pointing at a merged dataset_root."""
    payload: dict[str, Any] = {
        "dataset_mode": "reuse_validated",
        "dataset_root": str(dataset_root),
        "selected_methods": list(models),
        "models": list(models),
        "output_root": str(result_dir),
        "system_label": system_label,
        "allow_regenerate_siesta": False,
    }
    if epochs is not None:
        payload["epochs"] = int(epochs)
    return payload


def _split_selected_ids(selected_ids: list[str]) -> tuple[list[str], list[str]]:
    small = [sid[len(_SMALL_PREFIX):] for sid in selected_ids if sid.startswith(_SMALL_PREFIX)]
    large = [sid[len(_LARGE_PREFIX):] for sid in selected_ids if sid.startswith(_LARGE_PREFIX)]
    return small, large


def run_mixing_sweep(
    small_by_size: dict[int, str | Path],
    large_by_size: dict[int, str | Path],
    output_root: str | Path,
    *,
    sizes: list[int] | None = None,
    modes: tuple[str, ...] = DEFAULT_MODES,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = 0,
    models: tuple[str, ...] = ("graph2mat", "deeph"),
    epochs: int | None = None,
    system_label: str = "graphene",
    dry_run: bool = True,
    launch_fn: LaunchFn | None = None,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Materialize + (optionally) train one merged dataset per permutation.

    In ``dry_run`` mode nothing is written: the returned summary lists the
    permutations and the merged-dataset dirs that *would* be created. Otherwise
    each permutation is materialized and, if ``launch_fn`` is given, trained by
    it; per-model ``h_mae_eV`` records are collected for the MAE-vs-size plot.
    """
    output_root = Path(output_root)
    small_by_size = {int(k): Path(v) for k, v in small_by_size.items()}
    large_by_size = {int(k): Path(v) for k, v in large_by_size.items()}

    plan = plan_mixing_sweep_from_roots(
        small_by_size, large_by_size, sizes=sizes, modes=modes, ratios=ratios, seed=seed
    )

    records: list[dict[str, Any]] = []
    permutation_results: list[dict[str, Any]] = []

    for size in plan["sizes"]:
        if size not in small_by_size or size not in large_by_size:
            continue
        small_root = small_by_size[size]
        large_root = large_by_size[size]
        small_samples = [
            {"id": f"{_SMALL_PREFIX}{s.sample_id}"} for s in read_dataset_samples(small_root)
        ]
        large_samples = [
            {"id": f"{_LARGE_PREFIX}{s.sample_id}"} for s in read_dataset_samples(large_root)
        ]
        for mode in modes:
            manifest = make_mixed_dataset_manifest(
                small_samples, large_samples, ratios=ratios, mode=mode, seed=seed
            )
            for part in manifest["partitions"]:
                ratio = float(part["ratio"])
                merged_dir = output_root / f"size{size}_{mode}_{_ratio_slug(ratio)}"
                selected_small, selected_large = _split_selected_ids(part["selected_ids"])
                entry: dict[str, Any] = {
                    "size": size,
                    "mode": mode,
                    "ratio": ratio,
                    "total_size": int(part["n_selected"]),
                    "n_small_selected": len(selected_small),
                    "n_large_selected": len(selected_large),
                    "output_root": str(merged_dir),
                }
                if dry_run:
                    entry["status"] = "planned"
                    permutation_results.append(entry)
                    continue

                materialize_summary = materialize_mixed_dataset(
                    small_root,
                    large_root,
                    selected_small_ids=selected_small,
                    selected_large_ids=selected_large,
                    output_root=merged_dir,
                    seed=seed,
                )
                entry["materialize"] = materialize_summary
                if launch_fn is not None:
                    payload = _build_runner_payload(
                        merged_dir,
                        merged_dir / "training",
                        models,
                        epochs=epochs,
                        system_label=system_label,
                    )
                    launch_result = launch_fn(payload)
                    entry["launch"] = launch_result
                    metrics = (launch_result or {}).get("metrics") or {}
                    for model, model_metrics in metrics.items():
                        h_mae = (model_metrics or {}).get("h_mae_eV")
                        if h_mae is None:
                            continue
                        records.append(
                            {
                                "size": size,
                                "mode": mode,
                                "ratio": ratio,
                                "total_size": int(part["n_selected"]),
                                "model": model,
                                "h_mae_eV": float(h_mae),
                                "output_root": str(merged_dir),
                            }
                        )
                    entry["status"] = "trained"
                else:
                    entry["status"] = "materialized"
                permutation_results.append(entry)
                if progress_fn is not None:
                    progress_fn(dict(entry))

    summary = {
        "schema": "ml_vs_siesta_mixing_sweep_summary_v1",
        "dry_run": dry_run,
        "modes": list(modes),
        "ratios": list(ratios),
        "seed": seed,
        "models": list(models),
        "plan_warnings": plan["warnings"],
        "n_permutations": len(permutation_results),
        "permutations": permutation_results,
        "records": records,
    }
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "mixing_sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary
