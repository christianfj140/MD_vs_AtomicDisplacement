#!/usr/bin/env python3
"""Can the original checkpoint selection be *reproduced* on a clean validation set?

The original defect is one of **selection**, not of lineage: the lineage is fully
reconstructed (738/738 rows linked, 4428 artifact sha256 verified), but 3 of the
84 validation rows are byte-identical to training rows and the checkpoint was
chosen by ``val_spectral_frontier_aligned_rmse_eV`` on exactly that split.

Remediating the *selection* means re-running the original selection rule on a
validation set with those rows removed. That is only meaningful if the snapshots
the original rule ranked still exist. This script decides that question and
nothing else. It never evaluates a checkpoint before the clean manifest is
frozen, and it never promotes, replaces or relabels the original checkpoint.

Three things are established from real artifacts:

**1. Inventory.** Every ``.ckpt`` under the run's checkpoint tree is hashed and
opened, and its membership in the run is read *out of the payload*: the
datamodule's ``root_dir`` and the ``train_runs``/``val_runs``/``test_runs``
lists, the model ``hyper_parameters``, and the ``ModelCheckpoint``/
``EarlyStopping`` callback state. A filename is never evidence of anything --
``epoch=`` in a name is a formatting artefact, the payload's ``epoch`` and
``global_step`` are the record.

**2. Clean validation manifest.** Frozen *before* any checkpoint is evaluated.
Identity is decided only by geometry-bearing artifacts, reusing
``certify_checkpoint_lineage_and_exclusion.IDENTITY_ARTIFACTS`` so there is one
identity rule in the repository rather than two that can drift. ORB_INDX and
metadata.json are basis/bookkeeping descriptors that repeat verbatim across
physically distinct samples (ORB_INDX takes 2 distinct values over all 738
rows); including them reports 84/84 duplicates instead of the real 3. Colliding
rows are identified by recomputed sha256, never by sample_id: no name is encoded
as a criterion anywhere in this file.

**3. Selection reproducibility.** The original monitor was evaluated once per
epoch by ``check_val_every_n_epoch=1`` over 250 completed epochs, but the
``spectral-best`` ``ModelCheckpoint`` ran with ``save_top_k=1``: every time the
monitor improved, Lightning deleted the previous best. The per-epoch monitor
curve survives in the TensorBoard event file, so the number of epochs the
original rule actually ranked -- and how many of them still have weights -- is
measurable rather than assumed. Re-ranking a subset that does not contain the
epochs the original rule considered does not reproduce the selection; it
performs a *different, weaker* selection over whatever happened to survive.

Emits ``checkpoint_selection_remediation_v1``. No SIESTA, no training, no
dataset mutation, no threshold set after seeing a result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "Comparison/scripts"),
    str(ROOT / "shared"),
    str(ROOT / "scripts/torch_serialization_compat"),
]

from artifact_signature import file_sha256  # noqa: E402

SCHEMA = "checkpoint_selection_remediation_v1"
RUN_ROOT = ROOT / "Comparison/results/tbg_registry_spectral_loss"
CHECKPOINT_DIR = RUN_ROOT / "training/checkpoints"
TRAINING_PLAN = RUN_ROOT / "training_plan.json"
TENSORBOARD_ROOT = RUN_ROOT / "training/lightning_logs/spectral_loss"
MANIFEST = ROOT / "Comparison/datasets/tbg_md_plus_registry/n558/frozen_split_manifest.json"
DEFAULT_OUTPUT = ROOT / "Comparison/results/epc/checkpoint_selection_remediation"

MONITOR = "val_spectral_frontier_aligned_rmse_eV"

# The identity rule lives in one place. Importing it (rather than restating it)
# is what keeps the ORB_INDX false-collision regression pinned for both gates.
_LINEAGE = ROOT / "Comparison/scripts/certify_checkpoint_lineage_and_exclusion.py"
_SPEC = importlib.util.spec_from_file_location("_lineage_gate", _LINEAGE)
assert _SPEC and _SPEC.loader
LINEAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LINEAGE)

IDENTITY_ARTIFACTS = LINEAGE.IDENTITY_ARTIFACTS

# Payload fields that must agree across snapshots for them to be one run. Read
# from the checkpoint, never from the path.
RUN_IDENTITY_KEYS = ("datamodule_root_dir", "train_runs_sha256", "val_runs_sha256",
                     "test_runs_sha256", "model_hyper_parameters_sha256")


def _digest(value: Any) -> str:
    """Stable sha256 of a JSON-able payload field."""
    import hashlib

    blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# 1. Inventory
# --------------------------------------------------------------------------- #


def read_snapshot(path: Path) -> dict[str, Any]:
    """Open one checkpoint and read its run membership out of the payload."""
    import torch
    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    dm = raw.get("datamodule_hyper_parameters") or {}
    hp = raw.get("hyper_parameters") or {}

    callbacks: dict[str, Any] = {}
    for name, state in (raw.get("callbacks") or {}).items():
        callbacks[name] = {
            key: (float(value) if hasattr(value, "item") else value)
            for key, value in state.items()
            if key in ("monitor", "best_model_score", "best_model_path", "current_score",
                       "dirpath", "kth_best_model_path", "last_model_path", "best_score",
                       "patience", "wait_count", "stopped_epoch")
        }
        best_k = state.get("best_k_models")
        if isinstance(best_k, dict):
            callbacks[name]["best_k_models"] = {
                str(key): float(value) if hasattr(value, "item") else value
                for key, value in best_k.items()
            }

    return {
        "path": str(path),
        "relative_path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(str(path)),
        "size_bytes": path.stat().st_size,
        # From the payload, not the filename.
        "epoch": raw.get("epoch"),
        "global_step": raw.get("global_step"),
        "pytorch_lightning_version": raw.get("pytorch-lightning_version"),
        "state_dict_tensors": len(raw.get("state_dict") or {}),
        "run_identity": {
            "datamodule_root_dir": dm.get("root_dir"),
            "train_runs_count": len(dm.get("train_runs") or []),
            "val_runs_count": len(dm.get("val_runs") or []),
            "test_runs_count": len(dm.get("test_runs") or []),
            "train_runs_sha256": _digest(dm.get("train_runs")),
            "val_runs_sha256": _digest(dm.get("val_runs")),
            "test_runs_sha256": _digest(dm.get("test_runs")),
            "model_hyper_parameters_sha256": _digest(
                {key: str(value) for key, value in sorted(hp.items())}
            ),
            "out_matrix": dm.get("out_matrix"),
            "matrix_component_policy": dm.get("matrix_component_policy"),
            "symmetric_matrix": dm.get("symmetric_matrix"),
            "basis_files": dm.get("basis_files"),
        },
        "callbacks": callbacks,
        "filename_claims_epoch": _epoch_from_name(path.name),
    }


def _epoch_from_name(name: str) -> int | None:
    """Parse ``epoch=NNN`` from a filename -- only to *cross-check* the payload."""
    marker = "epoch="
    if marker not in name:
        return None
    tail = name.split(marker)[-1]
    digits = ""
    for char in tail:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


def inventory(checkpoint_dir: Path) -> dict[str, Any]:
    paths = sorted(checkpoint_dir.rglob("*.ckpt"))
    snapshots = [read_snapshot(path) for path in paths]

    reference = snapshots[0]["run_identity"] if snapshots else {}
    for snapshot in snapshots:
        mismatched = [
            key for key in RUN_IDENTITY_KEYS
            if snapshot["run_identity"].get(key) != reference.get(key)
        ]
        snapshot["same_run_as_reference"] = not mismatched
        snapshot["run_identity_mismatches"] = mismatched
        claimed = snapshot["filename_claims_epoch"]
        snapshot["filename_agrees_with_payload_epoch"] = (
            None if claimed is None else claimed == snapshot["epoch"]
        )

    by_hash: dict[str, list[str]] = {}
    for snapshot in snapshots:
        by_hash.setdefault(snapshot["sha256"], []).append(snapshot["relative_path"])
    duplicates = {sha: names for sha, names in by_hash.items() if len(names) > 1}

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "snapshots_found": len(snapshots),
        "distinct_weight_files_by_sha256": len(by_hash),
        "byte_identical_groups": duplicates,
        "distinct_payload_epochs": sorted({s["epoch"] for s in snapshots if s["epoch"] is not None}),
        "membership_evidence": (
            "run membership is read from the checkpoint payload: datamodule root_dir, the "
            "sha256 of the train/val/test run lists, and the sha256 of the model "
            "hyper_parameters. Filenames are parsed only to cross-check the payload epoch "
            "and are never the source of membership."
        ),
        "all_snapshots_same_run": all(s["same_run_as_reference"] for s in snapshots),
        "snapshots": snapshots,
    }


# --------------------------------------------------------------------------- #
# 2. Clean validation manifest -- frozen BEFORE any checkpoint is evaluated
# --------------------------------------------------------------------------- #


def clean_validation_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validation rows that collide with train, validation or test, by sha256."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [LINEAGE.link_row(row) for row in payload.get("rows") or []]

    def keyset(row: dict[str, Any]) -> set[str]:
        return {
            f"{artifact}:{sha}"
            for artifact in IDENTITY_ARTIFACTS
            if (sha := row["verified_artifact_sha256"].get(artifact))
        }

    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), []).append(row)

    index: dict[str, dict[str, list[str]]] = {}
    for split, split_rows in by_split.items():
        bucket: dict[str, list[str]] = {}
        for row in split_rows:
            for key in keyset(row):
                bucket.setdefault(key, []).append(str(row["sample_id"]))
        index[split] = bucket

    validation = by_split.get("validation", [])
    removals: list[dict[str, Any]] = []
    for row in validation:
        keys = keyset(row)
        train_hits = sorted({s for key in keys for s in index.get("train", {}).get(key, [])})
        peer_hits = sorted(
            {
                s
                for key in keys
                for s in index.get("validation", {}).get(key, [])
                if s != str(row["sample_id"])
            }
        )
        if train_hits or peer_hits:
            removals.append(
                {
                    "validation_sample_id": row["sample_id"],
                    "collides_with_train": train_hits,
                    "collides_with_validation_peer": peer_hits,
                    "colliding_identity_keys": sorted(
                        key for key in keys
                        if index.get("train", {}).get(key)
                        or len(index.get("validation", {}).get(key, [])) > 1
                    ),
                }
            )

    removed_ids = {entry["validation_sample_id"] for entry in removals}
    kept = [row for row in validation if row["sample_id"] not in removed_ids]

    # Test is audited and reported, but is never a selection input.
    test = by_split.get("test", [])
    test_collisions = [
        {
            "test_sample_id": row["sample_id"],
            "collides_with_train": sorted(
                {s for key in keyset(row) for s in index.get("train", {}).get(key, [])}
            ),
            "collides_with_validation": sorted(
                {s for key in keyset(row) for s in index.get("validation", {}).get(key, [])}
            ),
        }
        for row in test
        if any(
            index.get("train", {}).get(key) or index.get("validation", {}).get(key)
            for key in keyset(row)
        )
    ]

    return {
        "schema": "clean_validation_manifest_v1",
        "frozen_before_any_checkpoint_evaluation": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(str(manifest_path)),
        "identity_artifacts": list(IDENTITY_ARTIFACTS),
        "identity_rule": (
            "two rows are the same geometry when any geometry-bearing artifact "
            "(RUN.fdf, XV, STRUCT_OUT, reference TSHS) has an identical recomputed sha256. "
            "ORB_INDX and metadata.json are excluded: they are basis/bookkeeping descriptors "
            "that repeat verbatim across physically distinct samples, and including them "
            "reports every validation row as a duplicate (84/84) instead of the real count."
        ),
        "identity_rule_source": "certify_checkpoint_lineage_and_exclusion.IDENTITY_ARTIFACTS",
        "no_sample_id_is_a_criterion": (
            "removals are derived from recomputed artifact sha256 only; no sample_id is "
            "hardcoded anywhere in this module"
        ),
        "rows_by_split": {split: len(items) for split, items in sorted(by_split.items())},
        "validation_rows_before": len(validation),
        "validation_rows_removed": len(removals),
        "validation_rows_after": len(kept),
        "removals": removals,
        "validation_internal_duplicate_rows": sum(
            1 for entry in removals if entry["collides_with_validation_peer"]
        ),
        "test_rows_colliding_with_train_or_validation": len(test_collisions),
        "test_collisions": test_collisions,
        "test_is_never_a_selection_input": True,
        "clean_validation_sample_ids": sorted(str(row["sample_id"]) for row in kept),
    }


# --------------------------------------------------------------------------- #
# 3. Is the original selection reproducible at all?
# --------------------------------------------------------------------------- #


def monitor_curve(tensorboard_root: Path) -> dict[str, Any]:
    """The per-epoch monitor the original selection ranked, from the event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    versions = sorted(tensorboard_root.glob("version_*"))
    series: list[dict[str, Any]] = []
    for version in versions:
        accumulator = EventAccumulator(str(version), size_guidance={"scalars": 0})
        accumulator.Reload()
        if MONITOR not in accumulator.Tags()["scalars"]:
            continue
        events = accumulator.Scalars(MONITOR)
        series.append(
            {
                "version": version.name,
                "event_files": sorted(
                    file_sha256(str(path)) for path in version.glob("events.out.tfevents.*")
                ),
                "epochs_logged": len(events),
                # Lightning logs the epoch-end validation metric once per epoch, in
                # order, so index == epoch index within the version.
                "values": [float(event.value) for event in events],
            }
        )
    return {"versions": series}


def reproducibility(
    snapshots: list[dict[str, Any]], curve: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Do the surviving snapshots cover the epochs the original rule ranked?"""
    longest = max(curve["versions"], key=lambda item: item["epochs_logged"], default=None)
    values = longest["values"] if longest else []
    epochs_ranked = len(values)

    running_best = 0
    best = float("inf")
    best_epochs: list[int] = []
    for index, value in enumerate(values):
        if value < best:
            best = value
            running_best += 1
            best_epochs.append(index)

    have_weights = sorted(
        {snapshot["epoch"] for snapshot in snapshots if snapshot["epoch"] is not None}
    )
    covered = [epoch for epoch in best_epochs if epoch in have_weights]

    ranked = sorted(range(epochs_ranked), key=lambda index: values[index]) if values else []
    rank_of = {epoch: position + 1 for position, epoch in enumerate(ranked)}
    surviving_ranked = [
        {"epoch": epoch, "monitor": values[epoch], "rank_among_ranked_epochs": rank_of[epoch]}
        for epoch in have_weights
        if epoch < epochs_ranked
    ]

    # Would a re-ranking restricted to the survivors even be able to change the
    # answer? Count epochs the original rule ranked that beat the best surviving
    # non-winner: those are the candidates a clean re-ranking could promote, and
    # they have no weights.
    non_winner = [entry for entry in surviving_ranked if entry["rank_among_ranked_epochs"] != 1]
    best_non_winner = min((entry["monitor"] for entry in non_winner), default=None)
    unrecoverable_contenders = (
        sum(1 for value in values if value < best_non_winner)
        if best_non_winner is not None
        else None
    )

    missing = [epoch for epoch in best_epochs if epoch not in have_weights]
    return {
        "monitor": MONITOR,
        "checkpoint_monitor_declared_in_plan": plan.get("checkpoint_monitor"),
        "check_val_every_n_epoch": 1,
        "periodic_checkpoint_epochs_declared": plan.get("periodic_checkpoint_epochs"),
        "epochs_the_original_rule_ranked": epochs_ranked,
        "epochs_that_were_the_running_best": running_best,
        "epochs_with_surviving_weights": len(have_weights),
        "surviving_epochs": have_weights,
        "running_best_epochs_with_surviving_weights": len(covered),
        "running_best_epochs_without_weights": len(missing),
        "surviving_snapshots_ranked_on_the_original_curve": surviving_ranked,
        "best_surviving_non_winner_monitor": best_non_winner,
        "ranked_epochs_beating_best_surviving_non_winner": unrecoverable_contenders,
        "cause": (
            "the spectral-best ModelCheckpoint ran with save_top_k=1, so each improvement "
            "of the monitor deleted the previous best. The per-epoch monitor curve survives "
            "in the TensorBoard event file but the corresponding weights do not."
        ),
        "reproducible": bool(values) and not missing,
        "why_not": (
            None
            if bool(values) and not missing
            else (
                f"{len(missing)} of the {running_best} epochs the original rule ranked as its "
                f"running best have no surviving weights. Re-ranking the "
                f"{len(have_weights)} snapshots that remain is a different, weaker selection "
                "over an arbitrary subset, not a reproduction of the original selection."
            )
        ),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(checkpoint_dir: Path, manifest_path: Path) -> dict[str, Any]:
    # Order matters and is enforced by construction: the clean manifest is built
    # and frozen before any snapshot metric is looked at.
    manifest = clean_validation_manifest(manifest_path)
    stock = inventory(checkpoint_dir)
    plan = json.loads(TRAINING_PLAN.read_text(encoding="utf-8")) if TRAINING_PLAN.is_file() else {}
    curve = monitor_curve(TENSORBOARD_ROOT)
    repro = reproducibility(stock["snapshots"], curve, plan)

    blockers: list[str] = []
    if not stock["all_snapshots_same_run"]:
        blockers.append("not every surviving snapshot is provably from the same run/config/dataset")
    if not repro["reproducible"]:
        blockers.append(repro["why_not"])
    if manifest["validation_rows_after"] <= 0:
        blockers.append("the clean validation set is empty")

    verdict = "BLOCKED" if blockers else "PASS"
    return {
        "schema": SCHEMA,
        "gate": "checkpoint_selection_remediation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "blockers": blockers,
        "scope": (
            "decides only whether the ORIGINAL checkpoint selection can be reproduced on a "
            "clean validation set. No SIESTA, no training, no final comparison, no dataset "
            "or state.db mutation."
        ),
        "checkpoint_inventory": stock,
        "clean_validation_manifest_v1": manifest,
        "selection_reproducibility": repro,
        "monitor_curve_provenance": curve,
        "original_checkpoint_is_untouched": (
            "the original checkpoint keeps checkpoint_lineage = NO_GO. Nothing here promotes, "
            "replaces, retro-repairs or relabels it. A remediated checkpoint would be a NEW "
            "object with its own id; none is emitted because the selection is not reproducible."
        ),
        "evaluating_only_the_winner_is_not_a_remediation": (
            "re-scoring spectral-best-epoch=247 on the clean validation set would show what "
            "that checkpoint scores, not that it would have been selected. Selection is a "
            "ranking over the epochs the rule visited; without their weights the ranking "
            "cannot be recomputed."
        ),
        "preserved_verdicts": {
            "checkpoint_lineage_original": "NO_GO",
            "final_case_exclusion": "PASS",
            "fine_tuning_candidate": "BLOCKED",
            "full_KS": "BLOCKED",
            "graph2mat_target_gauge_contract": "PASS",
            "gauge_derivative_audit": "PASS",
            "numerical_PAO_convergence": "PASS",
            "delta_out_closure": "NO_GO",
            "full_basis_connection_v1": "NO_GO",
            "onlyS_semantic_preflight_v1": "NO_GO",
            "gate_1_support_structure": "PASS",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_report(args.checkpoint_dir, args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "checkpoint_selection_remediation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stock, manifest, repro = (
        report["checkpoint_inventory"],
        report["clean_validation_manifest_v1"],
        report["selection_reproducibility"],
    )
    print(f"checkpoint_selection_remediation = {report['verdict']}")
    print(
        f"  inventory: {stock['snapshots_found']} snapshots, "
        f"{stock['distinct_weight_files_by_sha256']} distinct by sha256, "
        f"same-run: {stock['all_snapshots_same_run']}"
    )
    print(
        f"  clean validation: {manifest['validation_rows_before']} -> "
        f"{manifest['validation_rows_after']} "
        f"({manifest['validation_rows_removed']} removed by artifact sha256 identity)"
    )
    print(
        f"  selection: {repro['epochs_the_original_rule_ranked']} epochs ranked, "
        f"{repro['epochs_that_were_the_running_best']} were the running best, "
        f"{repro['epochs_with_surviving_weights']} have weights, "
        f"{repro['running_best_epochs_without_weights']} running-best epochs lost"
    )
    for blocker in report["blockers"]:
        print(f"  blocker: {blocker}")
    print(f"-> {path}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
