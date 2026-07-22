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

import concurrent.futures
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


def _dataset_label(root: str | Path) -> str:
    try:
        label = _load_json(Path(root) / "material_provenance.json").get("label")
    except Exception:  # noqa: BLE001 - planner reports unreadable datasets separately
        label = None
    return str(label or Path(root).name).strip()


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
        source_system_label = _dataset_label(source)
        for target in targets:
            target_id = _dataset_id(target)
            entry: dict[str, Any] = {
                "source_root": str(source),
                "target_root": str(target),
                "source_id": source_id,
                "target_id": target_id,
                "source_system_label": source_system_label,
                "target_system_label": _dataset_label(target),
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
    perm: dict[str, Any], model: str, h_mae_eV: float, seed: int,
    relative_frobenius: float | None = None,
) -> dict[str, Any]:
    record = {
        "source_id": perm["source_id"],
        "target_id": perm["target_id"],
        "source_system_label": perm.get("source_system_label"),
        "target_system_label": perm.get("target_system_label"),
        "payload_id": perm["payload_id"],
        "source_n_snapshots": perm.get("source_n_snapshots"),
        "source_n_atoms": perm.get("source_n_atoms"),
        "target_n_atoms": perm.get("target_n_atoms"),
        "model": model,
        "seed": seed,
        "h_mae_eV": float(h_mae_eV),
        "output_root": perm.get("output_root"),
    }
    if relative_frobenius is not None:
        record["relative_frobenius"] = float(relative_frobenius)
    return record


def _artifacts_for_source(existing_artifacts: dict[str, Any], perm: dict[str, Any]) -> dict[str, Any]:
    for key in (perm.get("source_id"), Path(str(perm.get("source_root") or "")).name):
        artifacts = existing_artifacts.get(str(key))
        if isinstance(artifacts, dict):
            return dict(artifacts)
    raise ValueError(
        "predict_metrics requires existing_artifacts for source "
        f"{perm.get('source_id')} ({perm.get('source_root')})."
    )


def run_cross_structure_sweep(
    sources: list[str | Path],
    targets: list[str | Path],
    output_root: str | Path,
    *,
    pairs: list[tuple[str | Path, str | Path]] | None = None,
    models: tuple[str, ...] = ("graph2mat", "deeph"),
    epochs: int | None = None,
    hyperparams: dict[str, dict[str, Any]] | None = None,
    early_stopping: dict[str, Any] | None = None,
    existing_artifacts: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
    seed: int = 0,
    confirm_ghost_species_exemption: bool = False,
    confirm_incomplete_hamiltonian_semantics: bool = False,
    strict_dataset_validation: bool = True,
    action: str = "preview",
    dry_run: bool | None = None,
    launch_fn: LaunchFn | None = None,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Materialize + (optionally) train one composite dataset per compatible pair.

    ``action`` is ``preview`` (plan only), ``materialize`` (build composites),
    ``train`` (materialize + drive the real runner via ``launch_fn``), or
    ``predict_metrics`` (same runner path but with existing checkpoints). Incompatible
    pairs are skipped. Per-model ``h_mae_eV`` records (MAE vs training source) are
    emitted through ``progress_fn`` and persisted in ``cross_structure_sweep_summary.json``.
    ``training_sweep`` is not supported in cross-structure (enforced downstream by
    ``run_cross_structure_payload``).
    """
    output_root = Path(output_root)
    action = str(action or "preview").strip().lower()
    if action not in {"preview", "materialize", "train", "predict_metrics"}:
        raise ValueError("action must be one of: preview, materialize, train, predict_metrics.")
    if dry_run is None:
        dry_run = action == "preview"

    cross_schedule = str((performance or {}).get("cross_model_schedule") or "").strip().lower()
    if (
        cross_schedule == "deeph_then_graph2mat"
        and not dry_run
        and set(models) == {"graph2mat", "deeph"}
    ):
        stage_summaries: dict[str, dict[str, Any]] = {}
        for model, limit_key in (
            ("deeph", "max_parallel_deeph_training_jobs"),
            ("graph2mat", "max_parallel_graph2mat_training_jobs"),
        ):
            stage_performance = dict(performance or {})
            stage_performance["cross_model_schedule"] = "single_model"
            stage_performance["max_parallel_prediction_jobs"] = int(stage_performance.get(limit_key) or 1)
            stage_summaries[model] = run_cross_structure_sweep(
                sources,
                targets,
                output_root,
                pairs=pairs,
                models=(model,),
                epochs=epochs,
                hyperparams=hyperparams,
                early_stopping=early_stopping,
                existing_artifacts=existing_artifacts,
                performance=stage_performance,
                seed=seed,
                confirm_ghost_species_exemption=confirm_ghost_species_exemption,
                confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
                strict_dataset_validation=strict_dataset_validation,
                action=action,
                dry_run=dry_run,
                launch_fn=launch_fn,
                progress_fn=progress_fn,
            )
        records = [
            record
            for model in ("deeph", "graph2mat")
            for record in stage_summaries[model].get("records") or []
        ]
        by_payload: dict[str, dict[str, Any]] = {}
        for model in ("deeph", "graph2mat"):
            for perm in stage_summaries[model].get("permutations") or []:
                payload_id = str(perm.get("payload_id") or "")
                merged = by_payload.setdefault(payload_id, dict(perm))
                merged.setdefault("model_launches", {})[model] = perm.get("launch")
                merged.setdefault("model_statuses", {})[model] = perm.get("status")
        expected_status = "evaluated" if action == "predict_metrics" else "trained"
        for perm in by_payload.values():
            statuses = perm.get("model_statuses") or {}
            if all(statuses.get(model) == expected_status for model in ("deeph", "graph2mat")):
                perm["status"] = expected_status
            elif any(statuses.get(model) == "incompatible" for model in statuses):
                perm["status"] = "incompatible"
            else:
                perm["status"] = "failed"
        permutations = list(by_payload.values())
        summary = {
            "schema": "ml_vs_siesta_cross_structure_sweep_summary_v1",
            "dry_run": False,
            "action": action,
            "models": ["deeph", "graph2mat"],
            "model_schedule": "deeph_then_graph2mat",
            "model_parallelism": {
                "deeph": int((performance or {}).get("max_parallel_deeph_training_jobs") or 1),
                "graph2mat": int((performance or {}).get("max_parallel_graph2mat_training_jobs") or 1),
            },
            "seed": seed,
            "plan_warnings": stage_summaries["deeph"].get("plan_warnings") or [],
            "n_permutations": len(permutations),
            "n_trained": sum(perm.get("status") == "trained" for perm in permutations),
            "n_evaluated": sum(perm.get("status") == "evaluated" for perm in permutations),
            "n_partial": 0,
            "n_failed": sum(perm.get("status") == "failed" for perm in permutations),
            "n_incompatible": sum(perm.get("status") == "incompatible" for perm in permutations),
            "permutations": permutations,
            "records": records,
            "model_stage_summaries": stage_summaries,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "cross_structure_sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    if pairs:
        permutations: list[dict[str, Any]] = []
        warnings: list[str] = []
        for source, target in pairs:
            pair_plan = plan_cross_structure_sweep(
                [source],
                [target],
                confirm_ghost_species_exemption=confirm_ghost_species_exemption,
                confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
            )
            permutations.extend(pair_plan.get("permutations") or [])
            warnings.extend(pair_plan.get("warnings") or [])
        plan = {
            "schema": "ml_vs_siesta_cross_structure_sweep_plan_v1",
            "n_permutations": len(permutations),
            "n_compatible": sum(1 for p in permutations if p["status"] == "compatible"),
            "n_incompatible": sum(1 for p in permutations if p["status"] == "incompatible"),
            "permutations": permutations,
            "warnings": warnings,
        }
    else:
        plan = plan_cross_structure_sweep(
            sources,
            targets,
            confirm_ghost_species_exemption=confirm_ghost_species_exemption,
            confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
        )

    if action == "predict_metrics" and not plan["n_compatible"]:
        warnings = plan.get("warnings") or []
        detail = warnings[0] if warnings else "no readable source/target pairs"
        if len(warnings) > 1:
            detail += f" (+{len(warnings) - 1} more incompatible pairs)"
        raise ValueError(f"predict_metrics has no compatible pairs: {detail}")

    records: list[dict[str, Any]] = []
    permutation_results: list[dict[str, Any]] = []

    def run_permutation(raw_perm: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        perm = dict(raw_perm)
        local_records: list[dict[str, Any]] = []
        perm = dict(perm)
        pair_dir = output_root / perm["payload_id"]
        perm["output_root"] = str(pair_dir)
        if perm["status"] == "incompatible":
            return perm, local_records
        if dry_run:
            perm["status"] = "planned"
            return perm, local_records

        runner_payload = _runner_payload(
            models,
            epochs,
            performance,
            strict_dataset_validation,
            hyperparams=hyperparams,
            early_stopping=early_stopping,
            seed=seed,
        )
        if action == "predict_metrics":
            runner_payload["predict_metrics_only"] = True
            runner_payload["existing_model_artifacts"] = _artifacts_for_source(
                existing_artifacts or {}, perm
            )
        payload = {
            "action": action,
            "source_dataset_root": perm["source_root"],
            "target_dataset_root": perm["target_root"],
            "composite_dataset_root": str(pair_dir / "dataset"),
            "run_output_root": str(pair_dir / "training" / models[0] if len(models) == 1 else pair_dir / "training"),
            "confirm_ghost_species_exemption": confirm_ghost_species_exemption,
            "confirm_incomplete_hamiltonian_semantics": confirm_incomplete_hamiltonian_semantics,
            "runner_payload": runner_payload,
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
                local_records.append(
                    _record(
                        perm,
                        model,
                        h_mae,
                        seed,
                        (model_metrics or {}).get("relative_frobenius"),
                    )
                )
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
                perm["status"] = "evaluated" if action == "predict_metrics" else "trained"
        return perm, local_records

    parallel_jobs = max(
        1,
        int((performance or {}).get("max_parallel_prediction_jobs") or 1),
    )
    if parallel_jobs > 1 and not dry_run:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(parallel_jobs, len(plan["permutations"]) or 1),
            thread_name_prefix="cross-structure",
        ) as executor:
            results = executor.map(run_permutation, plan["permutations"])
            for perm, local_records in results:
                permutation_results.append(perm)
                records.extend(local_records)
                if progress_fn is not None:
                    progress_fn(dict(perm))
    else:
        for raw_perm in plan["permutations"]:
            perm, local_records = run_permutation(raw_perm)
            permutation_results.append(perm)
            records.extend(local_records)
            if progress_fn is not None:
                progress_fn(dict(perm))

    def _count(status: str) -> int:
        return sum(1 for entry in permutation_results if entry.get("status") == status)

    summary = {
        "schema": "ml_vs_siesta_cross_structure_sweep_summary_v1",
        "dry_run": dry_run,
        "action": action,
        "models": list(models),
        "max_parallel_jobs": parallel_jobs,
        "seed": seed,
        "plan_warnings": plan["warnings"],
        "n_permutations": len(permutation_results),
        "n_trained": _count("trained"),
        "n_evaluated": _count("evaluated"),
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
    strict_dataset_validation: bool = True,
    *,
    hyperparams: dict[str, dict[str, Any]] | None = None,
    early_stopping: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selected_methods": list(models),
        "models": list(models),
        "strict_dataset_validation": bool(strict_dataset_validation),
        "random_seed": int(seed),
    }
    graph2mat_overrides = dict((hyperparams or {}).get("graph2mat") or {})
    deeph_options = dict((hyperparams or {}).get("deeph") or {})
    # Assign, not setdefault: a payload hyperparam must never silently pin the
    # training seed across replicates of a multi-seed sweep.
    graph2mat_overrides["seed_everything"] = int(seed)
    deeph_options["seed"] = int(seed)
    if epochs is not None:
        epochs = int(epochs)
        payload["epochs"] = epochs
        graph2mat_overrides["max_epochs"] = epochs
        deeph_options["epochs"] = epochs
    if graph2mat_overrides:
        payload["graph2mat_overrides"] = graph2mat_overrides
    if deeph_options:
        payload["deeph"] = deeph_options
    if early_stopping:
        payload["early_stopping"] = dict(early_stopping)
    if performance:
        payload["performance"] = performance
    return payload


def aggregate_cross_structure_mae(records, *, by_seed: bool = False) -> dict[str, Any]:
    """Group ``h_mae_eV`` records into MAE-vs-training-source curves.

    Each record needs ``source_id``, ``target_id``, ``model``,
    ``source_n_snapshots`` (or ``source_n_atoms`` fallback) and ``h_mae_eV``.
    Curve key is ``(source_system_label, target_id, model)`` — one curve answers
    "how does the MAE on target Y scale for source structure X". Points are seed-aggregated
    (mean ± sample std), with fewer than ``MIN_SEEDS_FOR_CLAIMS`` seeds flagged
    ``exploratory``. Output shape matches the mixing aggregator so the same
    frontend renders it.

    When ``by_seed`` is True, seeds are NOT aggregated: the curve key also
    includes the seed and each effective ``payload_id`` is prefixed with
    ``seed{N}::`` so every (seed, pair, model) is a distinct selectable payload
    and point. Used by the vacancy selector to plot per-seed curves.
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
        raw_payload_id = str(record.get("payload_id") or f"{source_id}__to__{target_id}")
        source_system_label = str(record.get("source_system_label") or "source")
        target_system_label = str(record.get("target_system_label") or target_id)
        seed_value = record.get("seed")
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            seed = 0
        # Seed-aware mode makes each (seed, pair) a distinct selectable payload
        # by prefixing the id; the curve key gains the seed so seeds are not merged.
        payload_id = f"seed{seed}::{raw_payload_id}" if by_seed else raw_payload_id
        curve_key = (
            (source_system_label, target_id, model, seed)
            if by_seed
            else (source_system_label, target_id, model)
        )
        payload_entry = {
            "id": payload_id,
            "label": f"{source_id} → {target_id}",
            "source_id": source_id,
            "target_id": target_id,
            "source_system_label": source_system_label,
            "target_system_label": target_system_label,
            "source_n_snapshots": x,
            "output_root": record.get("output_root"),
        }
        if by_seed:
            payload_entry["seed"] = seed
        payloads.setdefault(payload_id, payload_entry)
        bucket = curves.setdefault(curve_key, {}).setdefault(
            payload_id,
            {
                "payload_id": payload_id,
                "source_id": source_id,
                "target_id": target_id,
                "source_system_label": source_system_label,
                "target_system_label": target_system_label,
                "seed": seed,
                "x": x,
                "values": [],
                "relative_frobenius_values": [],
                "seeds": set(),
            },
        )
        bucket["values"].append(mae)
        try:
            relative_frobenius = float(record["relative_frobenius"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            bucket["relative_frobenius_values"].append(relative_frobenius)
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
        relative_values = item["relative_frobenius_values"]
        relative_mean = sum(relative_values) / len(relative_values) if relative_values else None
        relative_std = (
            (sum((v - relative_mean) ** 2 for v in relative_values) / (len(relative_values) - 1)) ** 0.5
            if len(relative_values) > 1
            else None
        )
        return {
            "payload_id": item["payload_id"],
            "source_id": item["source_id"],
            "target_id": item["target_id"],
            "source_system_label": item["source_system_label"],
            "target_system_label": item["target_system_label"],
            "seed": item.get("seed"),
            "x": item["x"],
            "total_size": item["x"],  # frontend reuses total_size as the x fallback
            "actual_train_size": item["x"],
            "mae": mean,
            "mae_std": std,
            "relative_frobenius": relative_mean,
            "relative_frobenius_std": relative_std,
            "n_seeds": n_seeds,
            "exploratory": n_seeds < MIN_SEEDS_FOR_CLAIMS,
        }

    curve_list: list[dict[str, Any]] = []
    for curve_key, by_payload in sorted(curves.items(), key=lambda kv: kv[0]):
        source_system_label, target_id, model = curve_key[0], curve_key[1], curve_key[2]
        seed = curve_key[3] if len(curve_key) > 3 else None
        points = [
            _point(item)
            for item in sorted(by_payload.values(), key=lambda v: (v["x"], v["payload_id"]))
        ]
        curve = {
            "target_id": target_id,
            "source_system_label": source_system_label,
            "model": model,
            "label": f"{source_system_label} → {target_id} · {model}"
            + (f" · seed {seed}" if seed is not None else ""),
            "exploratory": any(p["exploratory"] for p in points),
            "points": points,
        }
        if seed is not None:
            curve["seed"] = seed
        curve_list.append(curve)
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

    # Seed-aware aggregation: seeds are not merged. Curves are keyed by seed too,
    # so seed 0 (s10+s50) and seed 1 (s10) are separate curves; each (seed, pair)
    # is a distinct payload with an id prefixed by seed{N}::.
    agg_seed = aggregate_cross_structure_mae(records, by_seed=True)
    assert agg_seed["n_curves"] == 2, agg_seed  # seed 0 curve, seed 1 curve
    ids = sorted(p["id"] for p in agg_seed["payloads"])
    assert ids == ["seed0::s10__to__t50", "seed0::s50__to__t50", "seed1::s10__to__t50"], ids
    assert all(p.get("seed") in (0, 1) for p in agg_seed["payloads"]), agg_seed["payloads"]
    seed0_curve = next(c for c in agg_seed["curves"] if c.get("seed") == 0)
    s10_pt = next(p for p in seed0_curve["points"] if p["source_id"] == "s10")
    assert abs(s10_pt["mae"] - 0.4) < 1e-9, s10_pt  # seed 0 only, not averaged with 0.6
    assert s10_pt["seed"] == 0 and s10_pt["n_seeds"] == 1
    print("cross_structure_sweep._demo OK")


if __name__ == "__main__":
    _demo()
