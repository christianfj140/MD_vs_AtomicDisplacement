"""Cross-structure dataset materialization for Graph2Mat/DeepH benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

from benchmark_manifest import (  # noqa: E402
    ARTIFACT_KEYS,
    FROZEN_SPLIT_SCHEMA,
    build_benchmark_dataset_manifest,
    canonical_sha256,
    file_sha256,
)
from joint_artifact_contract import validate_snapshot  # noqa: E402
from run_inventory import collect_run_inventory  # noqa: E402

from .mixed_dataset_materialize import (
    DatasetCompatibilityError,
    DatasetMaterializeError,
    DatasetSample,
    _copy_dataset_level_files,
    _link_or_copy,
    _load_json,
    _validate_safe_output_root,
    dataset_atom_count,
    read_dataset_samples,
    validate_datasets_compatible,
)

_SPLITS = ("train", "validation", "test")
_PROTECTED_RUNNER_KEYS = {
    "dataset_mode",
    "dataset_root",
    "output_root",
    "allow_regenerate_siesta",
}
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_path(value: str | Path, *, default: str | Path | None = None) -> Path:
    raw = str(value if value not in (None, "") else default or "")
    if not raw:
        raise DatasetMaterializeError("Missing required path.")
    expanded = raw.replace("${REPO_ROOT}", str(_REPO_ROOT))
    path = Path(os.path.expandvars(expanded)).expanduser()
    return path if path.is_absolute() else _REPO_ROOT / path


def _samples_by_split(dataset_root: Path) -> dict[str, list[DatasetSample]]:
    by_split = {split: [] for split in _SPLITS}
    for sample in read_dataset_samples(dataset_root):
        if sample.split in by_split:
            by_split[sample.split].append(sample)
    return by_split


def _require_non_empty(by_split: dict[str, list[DatasetSample]], split: str, label: str) -> None:
    if not by_split.get(split):
        raise DatasetMaterializeError(f"{label} dataset has no non-empty {split!r} split.")


def _split_hash(dataset_root: Path) -> str:
    payload = _load_json(dataset_root / "frozen_split_manifest.json")
    return str(payload.get("split_hash") or canonical_sha256(payload.get("rows") or []))


def _sample_identity(sample: DatasetSample) -> str:
    try:
        root = str(sample.source_root.resolve())
        sample_dir = str(sample.sample_dir.resolve())
    except OSError:
        root = str(sample.source_root)
        sample_dir = str(sample.sample_dir)
    return canonical_sha256(
        {
            "source_root": root,
            "sample_id": sample.sample_id,
            "sample_dir": sample_dir,
            "source_split": sample.split,
        }
    )


def _sample_identity_payload(sample: DatasetSample) -> dict[str, Any]:
    return {
        "identity": _sample_identity(sample),
        "source_root": str(sample.source_root),
        "sample_id": sample.sample_id,
        "sample_dir": str(sample.sample_dir),
        "source_split": sample.split,
    }


def _system_labels(samples: list[DatasetSample]) -> list[str]:
    return sorted({str(sample.system_label) for sample in samples if sample.system_label})


def _atom_counts(samples: list[DatasetSample], root: Path) -> list[int]:
    counts = {int(sample.n_atoms) for sample in samples if sample.n_atoms is not None}
    if not counts:
        fallback = dataset_atom_count(root)
        if fallback is not None:
            counts.add(int(fallback))
    return sorted(counts)


def _role_sample_id(role: str, split: str, sample_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)
    return f"{role}_{split}__{safe}"


def plan_cross_structure_dataset(
    source_dataset_root: str | Path,
    target_dataset_root: str | Path,
    *,
    confirm_ghost_species_exemption: bool = False,
    confirm_incomplete_hamiltonian_semantics: bool = False,
) -> dict[str, Any]:
    """Return a no-write plan for source train/val -> target test."""
    source = Path(source_dataset_root)
    target = Path(target_dataset_root)
    source_by_split = _samples_by_split(source)
    target_by_split = _samples_by_split(target)
    _require_non_empty(source_by_split, "train", "source")
    _require_non_empty(source_by_split, "validation", "source")
    _require_non_empty(target_by_split, "test", "target")
    compat = validate_datasets_compatible(
        source,
        target,
        confirm_ghost_species_exemption=confirm_ghost_species_exemption,
    )
    _validate_cross_structure_hashes(source, target, compat)
    hamiltonian_semantics = _hamiltonian_target_semantics(
        compat,
        source_root=source,
        target_root=target,
        confirmed=confirm_incomplete_hamiltonian_semantics,
    )
    if hamiltonian_semantics["mismatch_errors"] or (
        hamiltonian_semantics["incomplete_errors"] and not confirm_incomplete_hamiltonian_semantics
    ):
        raise DatasetCompatibilityError(
            "Incomplete Hamiltonian target semantics for cross-structure production run: "
            + "; ".join(hamiltonian_semantics["blocking_errors"])
            + ". Pass confirm_incomplete_hamiltonian_semantics=True only for preview/development runs."
        )
    selected = {
        "train": [("source", sample) for sample in source_by_split["train"]],
        "validation": [("source", sample) for sample in source_by_split["validation"]],
        "test": [("target", sample) for sample in target_by_split["test"]],
    }
    split_counts = {split: len(rows) for split, rows in selected.items()}
    all_samples = [sample for rows in selected.values() for _role, sample in rows]
    source_samples = source_by_split["train"] + source_by_split["validation"]
    target_samples = target_by_split["test"]
    return {
        "schema": "ml_vs_siesta_cross_structure_preview_v1",
        "evaluation_mode": "cross_structure",
        "source_dataset_root": str(source),
        "target_dataset_root": str(target),
        "source_split_hash": _split_hash(source),
        "target_split_hash": _split_hash(target),
        "split_counts": split_counts,
        "source_system_labels": _system_labels(source_samples),
        "target_system_labels": _system_labels(target_samples),
        "source_atom_counts": _atom_counts(source_samples, source),
        "target_atom_counts": _atom_counts(target_samples, target),
        "source_train_ids": [sample.sample_id for sample in source_by_split["train"]],
        "source_validation_ids": [sample.sample_id for sample in source_by_split["validation"]],
        "target_test_ids": [sample.sample_id for sample in target_by_split["test"]],
        "selected": selected,
        "compatibility": compat,
        "hamiltonian_target_semantics": hamiltonian_semantics,
        "leakage_check": _leakage_report(selected),
        "sample_identity_hash": canonical_sha256([_sample_identity(sample) for sample in all_samples]),
    }


def _leakage_report(selected: dict[str, list[tuple[str, DatasetSample]]]) -> dict[str, Any]:
    materialized_ids: dict[str, str] = {}
    identities: dict[str, str] = {}
    errors: list[str] = []
    target_in_train_val = 0
    source_in_test = 0
    identity_rows: list[dict[str, Any]] = []
    for split, rows in selected.items():
        for role, sample in rows:
            if role == "target" and split in {"train", "validation"}:
                target_in_train_val += 1
                errors.append(f"target sample in {split}: {sample.sample_id}")
            if role == "source" and split == "test":
                source_in_test += 1
                errors.append(f"source sample in test: {sample.sample_id}")
            mid = _role_sample_id(role, split, sample.sample_id)
            identity = _sample_identity(sample)
            identity_rows.append(
                {
                    "split": split,
                    "role": role,
                    "materialized_sample_id": mid,
                    **_sample_identity_payload(sample),
                }
            )
            if mid in materialized_ids:
                errors.append(f"materialized sample_id collision: {mid}")
            if identity in identities:
                errors.append(
                    f"canonical source artifact identity reused in {split} and {identities[identity]}: {sample.sample_id}"
                )
            materialized_ids[mid] = split
            identities[identity] = split
    return {
        "passed": not errors,
        "errors": errors,
        "target_samples_in_train_or_validation": target_in_train_val,
        "source_samples_in_test": source_in_test,
        "unique_materialized_sample_ids": len(materialized_ids),
        "unique_source_artifact_identities": len(identities),
        "materialized_sample_ids": materialized_ids,
        "source_artifact_identities": identity_rows,
    }


def _write_split_csvs(split_root: Path, split_rows: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = [
        "sample_id",
        "sample_dir",
        "split",
        "system_label",
        "source_root",
        "origin",
        "role",
        "source_sample_id",
        "source_split",
        "original_sample_id",
        "original_split",
        "n_atoms",
        "evaluation_mode",
        "transfer_direction",
        "artifact_validation_status",
        "source_artifact_identity",
        "linked_artifacts",
        "copied_artifacts",
    ]
    split_root.mkdir(parents=True, exist_ok=True)
    for split in _SPLITS:
        with (split_root / f"{split}_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows[split])


def _write_cross_material_provenance(
    source_root: Path,
    target_root: Path,
    output_root: Path,
    compat: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    source = _load_json(source_root / "material_provenance.json")
    target = _load_json(target_root / "material_provenance.json")
    label = f"cross_structure({source.get('label') or source_root.name}->{target.get('label') or target_root.name})"
    payload = {
        "schema": "ml_vs_siesta_cross_structure_material_provenance_v1",
        "profile": (
            "production"
            if source.get("profile") == target.get("profile") == "production"
            else "diagnostic"
        ),
        "material_source": "cross_structure_dataset",
        "heterogeneous_material_pool": True,
        "label": label,
        "source_dataset_root": str(source_root),
        "target_dataset_root": str(target_root),
        "train_validation_source": "source",
        "test_source": "target",
        "source_label": source.get("label") or source_root.name,
        "target_label": target.get("label") or target_root.name,
        "source_atom_counts": plan["source_atom_counts"],
        "target_atom_counts": plan["target_atom_counts"],
        "species": [
            entry
            for entry in (source.get("species") or [])
            if str(entry.get("label")) in set(compat.get("species") or [])
        ],
        "basis_file_sha256": compat.get("basis_file_sha256") or {},
        "pseudopotential_sha256": source.get("pseudopotential_sha256") or {},
        "fdf_sha256": hashlib.sha256(
            json.dumps(
                {"source": source.get("fdf_sha256"), "target": target.get("fdf_sha256")},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "fdf_sha256_semantics": "sha256_of_source_target_fdf_sha256_pair",
        "fdf_sha256_by_role": {
            "source": source.get("fdf_sha256"),
            "target": target.get("fdf_sha256"),
        },
        "compatibility": {
            "compatible": bool(compat.get("compatible")),
            "ghost_compatibility_status": compat.get("ghost_compatibility_status"),
        },
        "hamiltonian_target_semantics": plan["hamiltonian_target_semantics"],
    }
    (output_root / "material_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _transfer_direction(plan: dict[str, Any]) -> str:
    source = "_".join(f"{n}_atoms" for n in plan.get("source_atom_counts") or ["source"])
    target = "_".join(f"{n}_atoms" for n in plan.get("target_atom_counts") or ["target"])
    return f"{source}_to_{target}"


def _hash_for_species(mapping: dict[str, Any], label: str) -> str | None:
    if mapping.get(label) not in (None, ""):
        return str(mapping[label])
    for key, value in sorted(mapping.items()):
        if str(key).split(".", 1)[0] == label and value not in (None, ""):
            return str(value)
    return None


def _validate_cross_structure_hashes(source_root: Path, target_root: Path, compat: dict[str, Any]) -> None:
    source = _load_json(source_root / "material_provenance.json")
    target = _load_json(target_root / "material_provenance.json")
    errors: list[str] = []
    for label in [str(item) for item in compat.get("species") or []]:
        source_basis = _hash_for_species(source.get("basis_file_sha256") or {}, label)
        target_basis = _hash_for_species(target.get("basis_file_sha256") or {}, label)
        if not source_basis or not target_basis:
            errors.append(f"Missing orbital basis hash for real species {label}: source={bool(source_basis)} target={bool(target_basis)}")
        elif source_basis != target_basis:
            errors.append(f"Orbital basis hash differs for real species {label}: {source_basis[:12]}... vs {target_basis[:12]}...")

        source_pseudo = _hash_for_species(source.get("pseudopotential_sha256") or {}, label)
        target_pseudo = _hash_for_species(target.get("pseudopotential_sha256") or {}, label)
        if not source_pseudo or not target_pseudo:
            errors.append(
                f"Missing pseudopotential hash for real species {label}: source={bool(source_pseudo)} target={bool(target_pseudo)}"
            )
        elif source_pseudo != target_pseudo:
            errors.append(f"pseudopotential hash differs for real species {label}: {source_pseudo[:12]}... vs {target_pseudo[:12]}...")
    if errors:
        raise DatasetCompatibilityError("Cross-structure provenance is incomplete or incompatible: " + "; ".join(errors))


def _declared_hamiltonian_semantics(root: Path) -> dict[str, Any]:
    payload = _load_json(root / "material_provenance.json").get("hamiltonian_target_semantics")
    return payload if isinstance(payload, dict) else {}


def _semantic_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if payload.get(key) not in (None, ""):
            return payload[key]
    return None


def _canonical_h_only_policy(value: Any) -> str | None:
    text = str(value or "").strip()
    return "h_only" if text in {"h_only", "runner_forces_h_only"} else None


def _hamiltonian_target_semantics(
    compat: dict[str, Any],
    *,
    source_root: Path | None = None,
    target_root: Path | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    report = compat.get("compatibility_report") if isinstance(compat.get("compatibility_report"), dict) else {}
    source = _declared_hamiltonian_semantics(source_root) if source_root else {}
    target = _declared_hamiltonian_semantics(target_root) if target_root else {}
    source_policy = _semantic_value(source, "h_only_policy", "matrix_component_policy", "target_component_policy")
    target_policy = _semantic_value(target, "h_only_policy", "matrix_component_policy", "target_component_policy")
    source_components = _semantic_value(source, "matrix_component_count", "n_matrix_components")
    target_components = _semantic_value(target, "matrix_component_count", "n_matrix_components")
    source_repr = _semantic_value(source, "real_complex_representation", "matrix_representation")
    target_repr = _semantic_value(target, "real_complex_representation", "matrix_representation")
    source_spin = source.get("spin_semantics")
    target_spin = target.get("spin_semantics")
    incomplete: list[str] = []
    mismatches: list[str] = []
    source_policy_canonical = _canonical_h_only_policy(source_policy)
    target_policy_canonical = _canonical_h_only_policy(target_policy)
    if source_policy in (None, "") or target_policy in (None, ""):
        incomplete.append(f"H-only policy is not explicitly encoded on both datasets: {source_policy!r} vs {target_policy!r}")
    elif source_policy_canonical != "h_only" or target_policy_canonical != "h_only":
        mismatches.append(f"H-only policy is not h_only on both datasets: {source_policy!r} vs {target_policy!r}")
    try:
        component_pair = (int(source_components), int(target_components))
    except (TypeError, ValueError):
        component_pair = None
    if component_pair is None:
        incomplete.append(
            "matrix component count is not explicitly encoded as 1 on both datasets: "
            f"{source_components!r} vs {target_components!r}"
        )
    elif component_pair != (1, 1):
        mismatches.append(f"matrix component count differs from H-only 1 component: {source_components!r} vs {target_components!r}")
    if source_repr in (None, "") or target_repr in (None, ""):
        incomplete.append(
            "real/complex representation constraints are not explicitly encoded on both datasets: "
            f"{source_repr!r} vs {target_repr!r}"
        )
    elif source_repr not in {"real", "complex_supported"} or target_repr not in {"real", "complex_supported"}:
        mismatches.append(f"unsupported real/complex representation constraints: {source_repr!r} vs {target_repr!r}")
    elif source_repr != target_repr:
        mismatches.append(f"real/complex representation constraints differ: {source_repr!r} vs {target_repr!r}")
    if source_spin is not None and target_spin is not None and source_spin != target_spin:
        mismatches.append(f"explicit spin semantics differ: {source_spin!r} vs {target_spin!r}")
    errors = [*mismatches, *incomplete]
    warnings = ["Incomplete Hamiltonian target semantics accepted by explicit confirmation."] if incomplete and confirmed else []
    return {
        "h_only_policy": source_policy_canonical if source_policy_canonical == target_policy_canonical else "not_explicitly_encoded",
        "matrix_component_count": int(source_components) if component_pair == (1, 1) else "not_explicitly_encoded",
        "spin_semantics": {
            "status": "checked_by_dft_fingerprint",
            "source": "dataset_compatibility blocking scalar keys",
        },
        "real_complex_representation": source_repr if source_repr == target_repr else "not_explicitly_encoded",
        "compatibility_target_status": report.get("target_compatibility"),
        "source": source,
        "target": target,
        "blocking_errors": errors,
        "mismatch_errors": mismatches,
        "incomplete_errors": incomplete,
        "confirmed_incomplete_hamiltonian_semantics": bool(incomplete and confirmed and not mismatches),
        "warnings": warnings,
    }


def _copy_mode_summary(rows: list[dict[str, Any]], *, link_requested: bool) -> dict[str, Any]:
    linked = sum(int(row.get("linked_artifacts") or 0) for row in rows)
    copied = sum(int(row.get("copied_artifacts") or 0) for row in rows)
    return {
        "linked_artifacts": linked,
        "copied_artifacts": copied,
        "symlink_fallback_to_copy": bool(link_requested and copied > 0),
        "linked_or_copied": "mixed" if linked and copied else "linked" if linked else "copied",
    }


def _artifact_hashes_from_paths(paths: dict[str, str], path_map: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, final_path in sorted(paths.items()):
        source_path = Path(path_map.get(final_path, final_path))
        if source_path.is_file():
            hashes[key] = file_sha256(source_path)
    return hashes


def _replace_path_root(value: Any, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _replace_path_root(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_path_root(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        old = str(old_root)
        if value == old or value.startswith(old + "/"):
            return str(new_root) + value[len(old) :]
    return value


def _write_cross_benchmark_manifests(
    *,
    dataset_root: Path,
    split_root: Path,
    artifact_validation: dict[str, Any],
    material_provenance: dict[str, Any],
    path_map: dict[str, str],
) -> dict[str, Any]:
    frozen_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for split in _SPLITS:
        with (split_root / f"{split}_manifest.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            sample_dir = str(row.get("sample_dir") or "")
            snapshot = next(
                (item for item in artifact_validation.get("snapshots") or [] if item.get("snapshot_dir") == sample_dir),
                {},
            )
            present = dict(snapshot.get("present_artifacts") or {})
            artifact_paths = {
                output_key: present[input_key]
                for input_key, output_key in ARTIFACT_KEYS.items()
                if input_key in present
            }
            artifact_hashes = _artifact_hashes_from_paths(artifact_paths, path_map)
            sample_id = str(row.get("sample_id") or Path(sample_dir).name)
            valid = bool(snapshot.get("valid"))
            problems = list(snapshot.get("missing_required") or []) + list(snapshot.get("errors") or [])
            frozen_row: dict[str, Any] = {
                "sample_id": sample_id,
                "graph2mat_sample_id": sample_id,
                "deeph_sample_id": sample_id,
                "split": split,
                "sample_dir": sample_dir,
                "system_label": snapshot.get("system_label") or row.get("system_label"),
                "valid": valid,
                "validation_problems": problems,
                "artifact_paths": artifact_paths,
                "artifact_sha256": artifact_hashes,
            }
            for key, value in row.items():
                if key not in frozen_row and value not in (None, ""):
                    frozen_row[key] = value
            for artifact_key, artifact_path in sorted(artifact_paths.items()):
                frozen_row[f"{artifact_key}_path"] = artifact_path
                frozen_row[f"{artifact_key}_sha256"] = artifact_hashes.get(artifact_key, "")
            if not valid:
                warnings.append(f"{sample_id}: invalid joint artifacts: {problems}")
            frozen_rows.append(frozen_row)
    hash_rows = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "artifact_sha256": row["artifact_sha256"],
        }
        for row in frozen_rows
    ]
    frozen = {
        "schema": FROZEN_SPLIT_SCHEMA,
        "artifact_contract_version": "joint_graph2mat_deeph_artifact_contract_v1",
        "dataset_root": str(dataset_root),
        "split_root": str(dataset_root / "splits"),
        "split_hash": canonical_sha256(hash_rows),
        "split_counts": {split: sum(1 for row in frozen_rows if row.get("split") == split) for split in _SPLITS},
        "valid": not warnings and bool(frozen_rows),
        "warnings": warnings,
        "rows": frozen_rows,
    }
    live_root = split_root.parent
    live_validation = _replace_path_root(artifact_validation, dataset_root, live_root)
    benchmark = build_benchmark_dataset_manifest(
        live_root,
        artifact_validation=live_validation,
        frozen_split_manifest=frozen,
        material_provenance=material_provenance,
        generation_mode="cross_structure",
        strict_paper_ready_provenance=False,
    )
    benchmark = _replace_path_root(benchmark, live_root, dataset_root)
    if not benchmark["benchmark_ready"]:
        raise RuntimeError("Benchmark dataset manifest is not valid; refusing to freeze dataset for training.")
    (split_root.parent / "frozen_split_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (split_root.parent / "benchmark_dataset_manifest.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frozen


def _leakage_report_from_frozen_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    materialized_ids: dict[str, str] = {}
    identities: dict[str, str] = {}
    target_in_train_val = 0
    source_in_test = 0
    identity_rows: list[dict[str, Any]] = []
    for row in rows:
        split = str(row.get("split") or "")
        role = str(row.get("role") or row.get("origin") or "")
        sample_id = str(row.get("sample_id") or "")
        identity = str(row.get("source_artifact_identity") or "")
        original_id = str(row.get("original_sample_id") or row.get("source_sample_id") or "")
        original_split = str(row.get("original_split") or row.get("source_split") or "")
        source_root = str(row.get("source_root") or "")
        if split not in _SPLITS:
            errors.append(f"unknown split in frozen row: {split!r}")
            continue
        if not sample_id:
            errors.append(f"missing materialized sample_id in {split} row")
        if role not in {"source", "target"}:
            errors.append(f"missing or invalid role in {split} row: {role!r}")
        if not source_root:
            errors.append(f"missing source_root in {split} row: {sample_id}")
        if not original_id:
            errors.append(f"missing original/source sample_id in {split} row: {sample_id}")
        if not original_split:
            errors.append(f"missing original/source split in {split} row: {sample_id}")
        if not identity:
            errors.append(f"missing source_artifact_identity in {split} row: {sample_id}")
        if role == "target" and split in {"train", "validation"}:
            target_in_train_val += 1
            errors.append(f"target sample in {split}: {sample_id}")
        if role == "source" and split == "test":
            source_in_test += 1
            errors.append(f"source sample in test: {sample_id}")
        if sample_id in materialized_ids:
            errors.append(f"materialized sample_id collision: {sample_id}")
        if identity and identity in identities:
            errors.append(f"canonical source artifact identity reused in {split} and {identities[identity]}: {sample_id}")
        materialized_ids[sample_id] = split
        if identity:
            identities[identity] = split
        identity_rows.append(
            {
                "split": split,
                "role": role,
                "materialized_sample_id": sample_id,
                "identity": identity,
                "source_root": row.get("source_root"),
                "sample_id": original_id,
                "sample_dir": row.get("sample_dir"),
                "source_split": original_split,
            }
        )
    return {
        "passed": not errors,
        "errors": errors,
        "target_samples_in_train_or_validation": target_in_train_val,
        "source_samples_in_test": source_in_test,
        "unique_materialized_sample_ids": len(materialized_ids),
        "unique_source_artifact_identities": len(identities),
        "materialized_sample_ids": materialized_ids,
        "source_artifact_identities": identity_rows,
    }


def _cross_structure_metadata_from_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_mode": "cross_structure",
        "transfer_direction": provenance.get("transfer_direction"),
        "source_atom_counts": provenance.get("source_atom_counts") or [],
        "target_atom_counts": provenance.get("target_atom_counts") or [],
        "source_system_labels": provenance.get("source_system_labels") or [],
        "target_system_labels": provenance.get("target_system_labels") or [],
        "source_split_hash": provenance.get("source_split_hash"),
        "target_split_hash": provenance.get("target_split_hash"),
        "composite_split_hash": provenance.get("composite_split_hash"),
    }


def _validate_existing_cross_structure_dataset(
    output_root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    provenance = _load_json(output_root / "cross_structure_dataset_provenance.json")
    benchmark = _load_json(output_root / "benchmark_dataset_manifest.json")
    frozen = _load_json(output_root / "frozen_split_manifest.json")
    artifact_validation = _load_json(output_root / "artifact_validation.json")
    if provenance.get("schema") != "ml_vs_siesta_cross_structure_dataset_provenance_v1":
        raise DatasetMaterializeError(f"Existing composite is not a cross-structure dataset: {output_root}")
    expected = {
        "source_dataset_root": str(Path(plan["source_dataset_root"])),
        "target_dataset_root": str(Path(plan["target_dataset_root"])),
        "source_split_hash": plan["source_split_hash"],
        "target_split_hash": plan["target_split_hash"],
    }
    mismatches = [key for key, value in expected.items() if str(provenance.get(key)) != str(value)]
    if mismatches:
        raise DatasetMaterializeError(
            f"Existing cross-structure dataset does not match requested source/target: {', '.join(mismatches)}"
        )
    if not benchmark.get("benchmark_ready"):
        raise DatasetMaterializeError(f"Existing cross-structure dataset is not benchmark_ready: {output_root}")
    if not frozen.get("valid") or not artifact_validation.get("valid"):
        raise DatasetMaterializeError(f"Existing cross-structure dataset manifests are invalid: {output_root}")
    leakage = _leakage_report_from_frozen_rows(frozen.get("rows") or [])
    if not leakage.get("passed"):
        raise DatasetMaterializeError(
            f"Existing cross-structure dataset leakage check failed: {output_root}: "
            + "; ".join(leakage.get("errors") or [])
        )
    if dict(frozen.get("split_counts") or {}) != dict(plan["split_counts"]):
        raise DatasetMaterializeError("Existing cross-structure split counts do not match requested source/target plan.")
    target_ids = [
        str(row.get("original_sample_id") or row.get("source_sample_id") or "")
        for row in frozen.get("rows") or []
        if row.get("split") == "test"
    ]
    if target_ids != list(plan["target_test_ids"]):
        raise DatasetMaterializeError("Existing cross-structure target test IDs do not match requested target split.")
    if str(provenance.get("composite_split_hash")) != str(frozen.get("split_hash")):
        raise DatasetMaterializeError("Existing cross-structure composite split hash is stale.")
    return {
        "output_root": str(output_root),
        "split_counts": dict(frozen.get("split_counts") or plan["split_counts"]),
        "evaluation_mode": "cross_structure",
        "transfer_direction": provenance.get("transfer_direction") or _transfer_direction(plan),
        "compatibility": plan["compatibility"],
        "leakage_check": leakage,
        "cross_structure_metadata": _cross_structure_metadata_from_provenance(provenance),
        "frozen_split_hash": frozen.get("split_hash"),
        "reused_existing": True,
    }


def materialize_or_reuse_cross_structure_dataset(
    source_dataset_root: str | Path,
    target_dataset_root: str | Path,
    output_root: str | Path,
    *,
    link: bool = True,
    overwrite: bool = False,
    confirm_ghost_species_exemption: bool = False,
    confirm_incomplete_hamiltonian_semantics: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_root)
    plan = plan_cross_structure_dataset(
        Path(source_dataset_root),
        Path(target_dataset_root),
        confirm_ghost_species_exemption=confirm_ghost_species_exemption,
        confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
    )
    if output_path.exists() and not overwrite:
        return _validate_existing_cross_structure_dataset(output_path, plan)
    return materialize_cross_structure_dataset(
        source_dataset_root,
        target_dataset_root,
        output_path,
        link=link,
        overwrite=overwrite,
        confirm_ghost_species_exemption=confirm_ghost_species_exemption,
        confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
    )


def materialize_cross_structure_dataset(
    source_dataset_root: str | Path,
    target_dataset_root: str | Path,
    output_root: str | Path,
    *,
    link: bool = True,
    overwrite: bool = False,
    confirm_ghost_species_exemption: bool = False,
    confirm_incomplete_hamiltonian_semantics: bool = False,
) -> dict[str, Any]:
    """Build a runner-ready dataset: source train/validation, target test."""
    source_root = Path(source_dataset_root)
    target_root = Path(target_dataset_root)
    output_root = Path(output_root)
    _validate_safe_output_root(output_root)
    if output_root.exists() and not overwrite:
        raise DatasetMaterializeError(
            f"output_root already exists: {output_root}. Pass overwrite=True to replace it."
        )
    plan = plan_cross_structure_dataset(
        source_root,
        target_root,
        confirm_ghost_species_exemption=confirm_ghost_species_exemption,
        confirm_incomplete_hamiltonian_semantics=confirm_incomplete_hamiltonian_semantics,
    )
    if not plan["leakage_check"]["passed"]:
        raise DatasetMaterializeError("Cross-structure leakage check failed before materialization.")

    partial_root = output_root.with_name(f"{output_root.name}.partial-{uuid.uuid4().hex[:8]}")
    if partial_root.exists():
        shutil.rmtree(partial_root)
    partial_root.mkdir(parents=True, exist_ok=True)
    try:
        split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in _SPLITS}
        validation_snapshots: list[dict[str, Any]] = []
        artifact_path_map: dict[str, str] = {}
        transfer = _transfer_direction(plan)
        for split in _SPLITS:
            for idx, (role, sample) in enumerate(plan["selected"][split]):
                materialized_id = _role_sample_id(role, split, sample.sample_id)
                dest = partial_root / "splits" / split / str(idx)
                final_dest = output_root / "splits" / split / str(idx)
                dest.mkdir(parents=True, exist_ok=True)
                linked_count = 0
                copied_count = 0
                for artifact in sorted(path for path in sample.sample_dir.iterdir() if path.is_file()):
                    if artifact.name == "ML_prediction.HSX":
                        continue
                    before = dest / artifact.name
                    _link_or_copy(artifact, before, link=link)
                    if before.is_symlink():
                        linked_count += 1
                    else:
                        copied_count += 1
                result = validate_snapshot(dest)
                status = "valid" if result.valid else "invalid"
                final_present_artifacts = {}
                for key, value in result.present_artifacts.items():
                    final_value = str(final_dest / Path(value).name)
                    final_present_artifacts[key] = final_value
                    artifact_path_map[final_value] = str(value)
                validation_snapshots.append(
                    {
                        "snapshot_dir": str(final_dest),
                        "system_label": result.system_label,
                        "valid": bool(result.valid),
                        "repair_required": False,
                        "missing_required": list(result.missing_required),
                        "errors": list(result.errors),
                        "warnings": list(result.warnings),
                        "present_artifacts": final_present_artifacts,
                    }
                )
                split_rows[split].append(
                    {
                        "sample_id": materialized_id,
                        "sample_dir": str(final_dest),
                        "split": split,
                        "system_label": result.system_label or sample.system_label or "",
                        "source_root": str(sample.source_root),
                        "origin": role,
                        "role": role,
                        "source_sample_id": sample.sample_id,
                        "source_split": sample.split or "",
                        "original_sample_id": sample.sample_id,
                        "original_split": sample.split or "",
                        "n_atoms": sample.n_atoms if sample.n_atoms is not None else "",
                        "evaluation_mode": "cross_structure",
                        "transfer_direction": transfer,
                        "artifact_validation_status": status,
                        "source_artifact_identity": _sample_identity(sample),
                        "linked_artifacts": linked_count,
                        "copied_artifacts": copied_count,
                    }
                )

        _write_split_csvs(partial_root / "splits", split_rows)
        _copy_dataset_level_files(source_root, target_root, partial_root, plan["compatibility"])
        _write_cross_material_provenance(source_root, target_root, partial_root, plan["compatibility"], plan)
        report = plan["compatibility"].get("compatibility_report")
        if report:
            (partial_root / "dataset_compatibility_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        all_valid = all(s["valid"] for s in validation_snapshots)
        artifact_validation = {
            "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
            "valid": all_valid,
            "warnings": [] if all_valid else ["some cross-structure snapshots failed validation"],
            "snapshots": validation_snapshots,
        }
        if not all_valid:
            failed = [s for s in validation_snapshots if not s["valid"]]
            raise DatasetMaterializeError(
                f"{len(failed)} cross-structure snapshot(s) failed artifact validation; dataset was not materialized."
            )
        (partial_root / "artifact_validation.json").write_text(
            json.dumps(
                artifact_validation,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        material_provenance = _load_json(partial_root / "material_provenance.json")
        frozen = _write_cross_benchmark_manifests(
            dataset_root=output_root,
            split_root=partial_root / "splits",
            artifact_validation=artifact_validation,
            material_provenance=material_provenance,
            path_map=artifact_path_map,
        )
        leakage = plan["leakage_check"]
        flat_rows = [row for rows in split_rows.values() for row in rows]
        copy_mode = _copy_mode_summary(flat_rows, link_requested=bool(link))
        cross_metadata = {
            "evaluation_mode": "cross_structure",
            "transfer_direction": transfer,
            "source_atom_counts": plan["source_atom_counts"],
            "target_atom_counts": plan["target_atom_counts"],
            "source_system_labels": plan["source_system_labels"],
            "target_system_labels": plan["target_system_labels"],
            "source_split_hash": plan["source_split_hash"],
            "target_split_hash": plan["target_split_hash"],
            "composite_split_hash": frozen.get("split_hash"),
        }
        provenance = {
            "schema": "ml_vs_siesta_cross_structure_dataset_provenance_v1",
            "evaluation_mode": "cross_structure",
            "evaluation_scope": "target_structure_only",
            "validation_scope": "source_structure_only",
            "source_dataset_root": str(source_root),
            "target_dataset_root": str(target_root),
            "source_split_hash": plan["source_split_hash"],
            "target_split_hash": plan["target_split_hash"],
            "composite_split_hash": frozen.get("split_hash"),
            "source_system_labels": plan["source_system_labels"],
            "target_system_labels": plan["target_system_labels"],
            "source_atom_counts": plan["source_atom_counts"],
            "target_atom_counts": plan["target_atom_counts"],
            "train_count": plan["split_counts"]["train"],
            "validation_count": plan["split_counts"]["validation"],
            "test_count": plan["split_counts"]["test"],
            "source_train_ids": plan["source_train_ids"],
            "source_validation_ids": plan["source_validation_ids"],
            "target_test_ids": plan["target_test_ids"],
            "transfer_direction": transfer,
            "compatibility": plan["compatibility"],
            "leakage_check": leakage,
            "run_inventory": collect_run_inventory(),
            "link_requested": bool(link),
            "materialization": {
                **copy_mode,
                "split_rows": split_rows,
            },
            "hamiltonian_target_semantics": plan["hamiltonian_target_semantics"],
            "cross_structure_metadata": cross_metadata,
        }
        (partial_root / "cross_structure_dataset_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        partial_root.rename(output_root)
        return {
            "output_root": str(output_root),
            "split_counts": dict(plan["split_counts"]),
            "evaluation_mode": "cross_structure",
            "transfer_direction": transfer,
            "compatibility": plan["compatibility"],
            "leakage_check": leakage,
            "cross_structure_metadata": cross_metadata,
            "frozen_split_hash": frozen.get("split_hash"),
        }
    except Exception:
        shutil.rmtree(partial_root, ignore_errors=True)
        raise


def _build_runner_payload(payload: dict[str, Any], composite_root: Path, run_output_root: Path) -> dict[str, Any]:
    runner_payload = dict(payload.get("runner_payload") or {})
    conflicts = sorted(key for key in _PROTECTED_RUNNER_KEYS if key in runner_payload)
    if conflicts:
        raise DatasetMaterializeError(
            "runner_payload cannot override protected cross-structure fields: " + ", ".join(conflicts)
        )
    if "training_sweep" in runner_payload:
        raise DatasetMaterializeError(
            "cross-structure train action does not support runner_payload.training_sweep; "
            "use fixed runner settings so target-test metrics cannot drive search."
        )
    runner_payload.update(
        {
            "dataset_mode": "reuse_validated",
            "dataset_root": str(composite_root),
            "output_root": str(run_output_root),
            "allow_regenerate_siesta": False,
            "evaluation_mode": "cross_structure",
        }
    )
    runner_payload.setdefault("metric_fail_policy", "fail_closed")
    runner_payload.setdefault("allow_diagnostic_metrics", False)
    return runner_payload


def run_cross_structure_payload(
    payload: dict[str, Any],
    *,
    launch_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run preview/materialize/train for a cross-structure payload."""
    action = str(payload.get("action") or "preview").strip().lower()
    if action not in {"preview", "materialize", "train", "predict_metrics"}:
        raise DatasetMaterializeError("action must be one of: preview, materialize, train, predict_metrics.")
    source = _resolve_repo_path(payload.get("source_dataset_root"))
    target = _resolve_repo_path(payload.get("target_dataset_root"))
    composite = _resolve_repo_path(
        payload.get("composite_dataset_root"),
        default=Path("Comparison/results") / str(payload.get("case_id") or "cross_structure") / "dataset",
    )
    run_output = _resolve_repo_path(
        payload.get("run_output_root"),
        default=Path("Comparison/results") / str(payload.get("case_id") or "cross_structure") / "training",
    )
    confirm_ghost = bool(payload.get("confirm_ghost_species_exemption"))
    confirm_hamiltonian = bool(payload.get("confirm_incomplete_hamiltonian_semantics"))
    preview = plan_cross_structure_dataset(
        source,
        target,
        confirm_ghost_species_exemption=confirm_ghost,
        confirm_incomplete_hamiltonian_semantics=confirm_hamiltonian,
    )
    result = {"action": action, "preview": {key: value for key, value in preview.items() if key != "selected"}}
    if action == "preview":
        return result
    runner_payload = _build_runner_payload(payload, composite, run_output) if action in {"train", "predict_metrics"} else None
    if action == "predict_metrics" and runner_payload is not None:
        runner_payload["predict_metrics_only"] = True

    materialized = materialize_or_reuse_cross_structure_dataset(
        source,
        target,
        composite,
        link=bool(payload.get("link", True)),
        overwrite=bool(payload.get("overwrite", False)),
        confirm_ghost_species_exemption=confirm_ghost,
        confirm_incomplete_hamiltonian_semantics=confirm_hamiltonian,
    )
    result["materialized"] = materialized
    if action == "materialize":
        return result

    if runner_payload is not None and materialized.get("cross_structure_metadata"):
        runner_payload["cross_structure_metadata"] = materialized["cross_structure_metadata"]
    result["runner_payload"] = runner_payload
    if launch_fn is None:
        from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: PLC0415

        runner = Graph2MatDeepHBenchmarkRunner()
        launch_fn = runner.start
    launched = launch_fn(runner_payload)
    result["runner_result"] = launched
    return result
