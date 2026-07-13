"""Orchestrate a cross-structure sweep: source×target pairs -> MAE vs source.

A cross-structure *permutation* is a ``(source, target)`` pair (unlike the mixing
sweep, whose identity is ``(size, mode, ratio)``). ``plan_cross_structure_sweep``
is a pure planner: for every pair it runs :func:`plan_cross_structure_dataset` in
dry mode (no writes) and records the compatibility status; an incompatible pair
is flagged and skipped, never aborts the whole sweep.
``run_cross_structure_sweep`` materializes (reuses) each compatible pair and,
for ``action="train"``, drives the real runner via
:func:`run_cross_structure_payload`. It emits incremental records shaped exactly
like :func:`aggregate_cross_structure_mae` consumes (MAE vs training source).

This module never imports the heavy runner and never trains by itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .cross_structure_materialize import (
    materialize_or_reuse_cross_structure_dataset,
    plan_cross_structure_dataset,
    run_cross_structure_payload,
)
from .mixed_dataset_materialize import (
    DatasetCompatibilityError,
    DatasetMaterializeError,
    _load_json,
    dataset_atom_count,
    read_dataset_samples,
)

# Repo-wide statistical bar for claims (mirrors plot_mixing_mae_vs_size).
MIN_SEEDS_FOR_CLAIMS = 3

LaunchFn = Callable[[dict[str, Any]], dict[str, Any]]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _dataset_id(root: str | Path) -> str:
    """Stable id for a dataset, independent of tmp/absolute paths.

    The directory basename anchors the id so sibling datasets that share a
    provenance ``label`` (e.g. graphene_w90 sized ``iid10`` vs ``iid25``, both
    labelled "graphene") get distinct ids and never collapse to one sweep point.
    The label is prefixed when it adds information. Same principle as
    ``_mixing_payload_id``: the id must be the same across machines so
    plan/live/summary payloads merge instead of duplicating.
    """
    root = Path(root)
    try:
        label = _load_json(root / "material_provenance.json").get("label")
    except Exception:  # noqa: BLE001 - missing/broken provenance -> use dir name
        label = None
    name = root.name
    label = str(label or "").strip()
    if not label or label == name:
        return _slug(name)
    return _slug(f"{label}__{name}")


def _source_train_count(root: Path) -> int:
    """Number of training snapshots in ``root`` (train + validation)."""
    return sum(
        1
        for sample in read_dataset_samples(root)
        if str(sample.split) in {"train", "validation"}
    )


def plan_cross_structure_sweep(
    sources: list[str | Path],
    targets: list[str | Path],
    *,
    confirm_ghost_species_exemption: bool = False,
    confirm_incomplete_hamiltonian_semantics: bool = False,
) -> dict[str, Any]:
    """Enumerate ``(source, target)`` pairs with compatibility status.

    Pure: reuses :func:`plan_cross_structure_dataset` (dry, no writes). An
    incompatible pair is recorded with ``status="incompatible"`` and its reason,
    but never aborts the sweep. ``x`` for the plot is the source's real training
    snapshot count (``source_n_snapshots``), falling back to source atom count.
    """
    sources = [Path(s) for s in sources]
    targets = [Path(t) for t in targets]
    permutations: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source in sources:
        try:
            source_n_snapshots = _source_train_count(source)
            source_n_atoms = dataset_atom_count(source)
        except Exception as exc:  # noqa: BLE001 - unreadable source dataset
            warnings.append(f"source {source}: unreadable ({exc}); skipped")
            continue
        source_id = _dataset_id(source)
        for target in targets:
            target_id = _dataset_id(target)
            entry: dict[str, Any] = {
                "source_root": str(source),
                "target_root": str(target),
                "source_id": source_id,
                "target_id": target_id,
                "payload_id": f"{source_id}__to__{target_id}",
                "source_n_snapshots": source_n_snapshots,
                "source_n_atoms": source_n_atoms,
            }
            try:
                preview = plan_cross_structure_dataset(
                    source,
                    target,
                    confirm_ghost_species_exemption=confirm_ghost_species_exemption,
                    confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
                )
            except (DatasetCompatibilityError, DatasetMaterializeError) as exc:
                entry["status"] = "incompatible"
                entry["reason"] = str(exc)
                warnings.append(f"{source_id} -> {target_id}: incompatible ({exc})")
                permutations.append(entry)
                continue
            entry["status"] = "compatible"
            entry["target_n_atoms"] = (
                preview["target_atom_counts"][0] if preview["target_atom_counts"] else None
            )
            entry["split_counts"] = preview["split_counts"]
            permutations.append(entry)
    return {
        "schema": "ml_vs_siesta_cross_structure_sweep_plan_v1",
        "n_permutations": len(permutations),
        "n_compatible": sum(1 for p in permutations if p["status"] == "compatible"),
        "n_incompatible": sum(1 for p in permutations if p["status"] == "incompatible"),
        "permutations": permutations,
        "warnings": warnings,
    }


def _record(
    perm: dict[str, Any], model: str, h_mae_eV: float, seed: int
) -> dict[str, Any]:
    return {
        "source_id": perm["source_id"],
        "target_id": perm["target_id"],
        "payload_id": perm["payload_id"],
        "source_n_snapshots": perm.get("source_n_snapshots"),
        "source_n_atoms": perm.get("source_n_atoms"),
        "target_n_atoms": perm.get("target_n_atoms"),
        "model": model,
        "seed": seed,
        "h_mae_eV": float(h_mae_eV),
        "output_root": perm.get("output_root"),
    }


def run_cross_structure_sweep(
    sources: list[str | Path],
    targets: list[str | Path],
    output_root: str | Path,
    *,
    models: tuple[str, ...] = ("graph2mat", "deeph"),
    epochs: int | None = None,
    performance: dict[str, Any] | None = None,
    seed: int = 0,
    confirm_ghost_species_exemption: bool = False,
    confirm_incomplete_hamiltonian_semantics: bool = False,
    action: str = "preview",
    dry_run: bool | None = None,
    launch_fn: LaunchFn | None = None,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Materialize + (optionally) train one composite dataset per compatible pair.

    ``action`` is ``preview`` (plan only), ``materialize`` (build composites) or
    ``train`` (materialize + drive the real runner via ``launch_fn``). Incompatible
    pairs are skipped. Per-model ``h_mae_eV`` records (MAE vs training source) are
    emitted through ``progress_fn`` and persisted in ``cross_structure_sweep_summary.json``.
    ``training_sweep`` is not supported in cross-structure (enforced downstream by
    ``run_cross_structure_payload``).
    """
    output_root = Path(output_root)
    action = str(action or "preview").strip().lower()
    if action not in {"preview", "materialize", "train"}:
        raise ValueError("action must be one of: preview, materialize, train.")
    if dry_run is None:
        dry_run = action == "preview"

    plan = plan_cross_structure_sweep(
        sources,
        targets,
        confirm_ghost_species_exemption=confirm_ghost_species_exemption,
        confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
    )

    records: list[dict[str, Any]] = []
    permutation_results: list[dict[str, Any]] = []

    for perm in plan["permutations"]:
        perm = dict(perm)
        pair_dir = output_root / perm["payload_id"]
        perm["output_root"] = str(pair_dir)
        if perm["status"] == "incompatible":
            permutation_results.append(perm)
            if progress_fn is not None:
                progress_fn(dict(perm))
            continue
        if dry_run:
            perm["status"] = "planned"
            permutation_results.append(perm)
            if progress_fn is not None:
                progress_fn(dict(perm))
            continue

        payload = {
            "action": action,
            "source_dataset_root": perm["source_root"],
            "target_dataset_root": perm["target_root"],
            "composite_dataset_root": str(pair_dir / "dataset"),
            "run_output_root": str(pair_dir / "training"),
            "confirm_ghost_species_exemption": confirm_ghost_species_exemption,
            "confirm_incomplete_hamiltonian_semantics": confirm_incomplete_hamiltonian_semantics,
            "runner_payload": _runner_payload(models, epochs, performance),
        }
        if action == "materialize":
            # run_cross_structure_payload only materializes; no runner launch.
            result = run_cross_structure_payload(payload)
            perm["materialize"] = result.get("materialized")
            perm["status"] = "materialized"
        else:
            result = run_cross_structure_payload(payload, launch_fn=launch_fn)
            perm["materialize"] = result.get("materialized")
            launch_result = result.get("runner_result") or {}
            perm["launch"] = launch_result
            metrics = launch_result.get("metrics") or {}
            recorded: list[str] = []
            for model, model_metrics in metrics.items():
                h_mae = (model_metrics or {}).get("h_mae_eV")
                if h_mae is None:
                    continue
                recorded.append(model)
                records.append(_record(perm, model, h_mae, seed))
            launch_ok = launch_result.get("ok", True)
            if not launch_ok or not recorded:
                perm["status"] = "failed"
                perm["error"] = launch_result.get("error") or "no h_mae_eV produced"
            elif len(recorded) < len(models):
                perm["status"] = "partial"
                perm["error"] = "missing h_mae_eV for: " + ", ".join(
                    m for m in models if m not in recorded
                )
            else:
                perm["status"] = "trained"
        permutation_results.append(perm)
        if progress_fn is not None:
            progress_fn(dict(perm))

    def _count(status: str) -> int:
        return sum(1 for entry in permutation_results if entry.get("status") == status)

    summary = {
        "schema": "ml_vs_siesta_cross_structure_sweep_summary_v1",
        "dry_run": dry_run,
        "action": action,
        "models": list(models),
        "seed": seed,
        "plan_warnings": plan["warnings"],
        "n_permutations": len(permutation_results),
        "n_trained": _count("trained"),
        "n_partial": _count("partial"),
        "n_failed": _count("failed"),
        "n_incompatible": _count("incompatible"),
        "permutations": permutation_results,
        "records": records,
    }
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "cross_structure_sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary


def _runner_payload(
    models: tuple[str, ...],
    epochs: int | None,
    performance: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selected_methods": list(models),
        "models": list(models),
    }
    if epochs is not None:
        epochs = int(epochs)
        payload["epochs"] = epochs
        payload["graph2mat_overrides"] = {"max_epochs": epochs}
        payload["deeph"] = {"epochs": epochs}
    if performance:
        payload["performance"] = performance
    return payload


def aggregate_cross_structure_mae(records) -> dict[str, Any]:
    """Group ``h_mae_eV`` records into MAE-vs-training-source curves.

    Each record needs ``source_id``, ``target_id``, ``model``,
    ``source_n_snapshots`` (or ``source_n_atoms`` fallback) and ``h_mae_eV``.
    Curve key is ``(target_id, model)`` — one curve answers "how does the MAE on
    target Y change with which source X I trained on". Points are seed-aggregated
    (mean ± sample std), with fewer than ``MIN_SEEDS_FOR_CLAIMS`` seeds flagged
    ``exploratory``. Output shape matches the mixing aggregator so the same
    frontend renders it.
    """
    curves: dict[tuple, dict[str, dict[str, Any]]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            source_id = str(record["source_id"])
            target_id = str(record["target_id"])
            model = str(record["model"])
            mae = float(record["h_mae_eV"])
        except (KeyError, TypeError, ValueError):
            continue
        x = record.get("source_n_snapshots")
        if x in (None, ""):
            x = record.get("source_n_atoms")
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        payload_id = str(record.get("payload_id") or f"{source_id}__to__{target_id}")
        payloads.setdefault(
            payload_id,
            {
                "id": payload_id,
                "label": f"{source_id} → {target_id}",
                "source_id": source_id,
                "target_id": target_id,
                "source_n_snapshots": x,
                "output_root": record.get("output_root"),
            },
        )
        bucket = curves.setdefault((target_id, model), {}).setdefault(
            payload_id,
            {
                "payload_id": payload_id,
                "source_id": source_id,
                "target_id": target_id,
                "x": x,
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
        std = (
            (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            if len(values) > 1
            else None
        )
        n_seeds = len(item["seeds"]) if item["seeds"] else len(values)
        return {
            "payload_id": item["payload_id"],
            "source_id": item["source_id"],
            "target_id": item["target_id"],
            "x": item["x"],
            "total_size": item["x"],  # frontend reuses total_size as the x fallback
            "actual_train_size": item["x"],
            "mae": mean,
            "mae_std": std,
            "n_seeds": n_seeds,
            "exploratory": n_seeds < MIN_SEEDS_FOR_CLAIMS,
        }

    curve_list: list[dict[str, Any]] = []
    for (target_id, model), by_payload in sorted(curves.items(), key=lambda kv: kv[0]):
        points = [
            _point(item)
            for item in sorted(by_payload.values(), key=lambda v: (v["x"], v["payload_id"]))
        ]
        curve_list.append(
            {
                "target_id": target_id,
                "model": model,
                "label": f"→ {target_id} · {model}",
                "exploratory": any(p["exploratory"] for p in points),
                "points": points,
            }
        )
    exploratory = any(curve["exploratory"] for curve in curve_list)
    warnings: list[str] = []
    if exploratory and curve_list:
        warnings.append(
            f"curves with fewer than {MIN_SEEDS_FOR_CLAIMS} seeds per point are "
            "EXPLORATORY: seed-to-seed training variance is not resolved"
        )
    return {
        "schema": "ml_vs_siesta_cross_structure_mae_v1",
        "metric": "h_mae_eV",
        "x": "source_n_snapshots",
        "min_seeds_for_claims": MIN_SEEDS_FOR_CLAIMS,
        "exploratory": exploratory,
        "warnings": warnings,
        "n_curves": len(curve_list),
        "payloads": sorted(payloads.values(), key=lambda item: (item["target_id"], item["source_n_snapshots"], item["id"])),
        "curves": curve_list,
    }


def _demo() -> None:
    """Runnable check: planner marks incompatible without aborting; aggregator
    groups by (target, model) with x = source_n_snapshots."""
    # Aggregator: two sources -> one target, two seeds each -> mean/std.
    records = [
        {"source_id": "s10", "target_id": "t50", "payload_id": "s10__to__t50",
         "source_n_snapshots": 10, "model": "graph2mat", "seed": 0, "h_mae_eV": 0.4},
        {"source_id": "s10", "target_id": "t50", "payload_id": "s10__to__t50",
         "source_n_snapshots": 10, "model": "graph2mat", "seed": 1, "h_mae_eV": 0.6},
        {"source_id": "s50", "target_id": "t50", "payload_id": "s50__to__t50",
         "source_n_snapshots": 50, "model": "graph2mat", "seed": 0, "h_mae_eV": 0.2},
    ]
    agg = aggregate_cross_structure_mae(records)
    assert agg["n_curves"] == 1, agg
    curve = agg["curves"][0]
    assert curve["target_id"] == "t50" and curve["model"] == "graph2mat"
    xs = [p["x"] for p in curve["points"]]
    assert xs == [10, 50], xs
    p0 = curve["points"][0]
    assert abs(p0["mae"] - 0.5) < 1e-9 and p0["n_seeds"] == 2, p0
    assert len(agg["payloads"]) == 2
    print("cross_structure_sweep._demo OK")


if __name__ == "__main__":
    _demo()
