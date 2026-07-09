"""Deterministic manifests for joint Graph2Mat/DeepH benchmark datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from joint_artifact_contract import CONTRACT_NAME, validate_snapshot


MANIFEST_SCHEMA = "joint_graph2mat_deeph_benchmark_manifest_v1"
FROZEN_SPLIT_SCHEMA = "joint_graph2mat_deeph_frozen_split_manifest_v1"
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
ARTIFACT_KEYS = {
    "run_fdf": "run_fdf",
    "run_output": "run_output",
    "metadata": "metadata",
    "hsx": "reference_hsx",
    "tshs": "reference_tshs",
    "tsde": "reference_tsde",
    "struct_out": "struct_out",
    "xv": "xv",
    "orb_indx": "orb_indx",
}
SIESTA_FLAG_KEYS = (
    "SaveHS",
    "Save.HS",
    "TS.HS.Save",
    "TS.DE.Save",
    "XML.Write",
    "Write.OrbitalIndex",
)
SPIN_FLAG_KEYS = (
    "SpinPolarized",
    "FixSpin",
    "NonCollinearSpin",
)
ENVIRONMENT_PROVENANCE_KEYS = (
    "python_version",
    "platform",
    "executable",
    "package_versions",
    "conda_env_export_path",
    "pip_freeze_path",
    "container_image",
    "container_digest",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strip_fdf_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def fdf_directives(path: Path, keys: tuple[str, ...] = SIESTA_FLAG_KEYS) -> dict[str, str]:
    if not path.exists():
        return {}
    wanted = {key.lower(): key for key in keys}
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = _strip_fdf_comment(line)
        if not clean:
            continue
        parts = clean.split(None, 1)
        if not parts:
            continue
        canonical = wanted.get(parts[0].lower())
        if canonical:
            found[canonical] = parts[1].strip() if len(parts) > 1 else ""
    return {key: found[key] for key in keys if key in found}


def _artifact_hashes(paths: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, value in sorted(paths.items()):
        path = Path(value)
        if path.exists() and path.is_file():
            hashes[key] = file_sha256(path)
    return hashes


def _relative_or_absolute(path: str, root: Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        return str(candidate)


def _snapshot_artifacts(sample_dir: Path) -> tuple[dict[str, str], dict[str, str], str | None, bool, list[str]]:
    validation = validate_snapshot(sample_dir)
    artifact_paths = {
        output_key: validation.present_artifacts[input_key]
        for input_key, output_key in ARTIFACT_KEYS.items()
        if input_key in validation.present_artifacts
    }
    return (
        artifact_paths,
        _artifact_hashes(artifact_paths),
        validation.system_label,
        validation.valid,
        validation.missing_required + validation.errors,
    )


def split_manifest_rows(split_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "validation", "test"):
        for row in read_csv_rows(split_root / f"{split}_manifest.csv"):
            merged = dict(row)
            merged["split"] = split
            rows.append(merged)
    return sorted(
        rows,
        key=lambda row: (
            SPLIT_ORDER.get(str(row.get("split") or ""), 99),
            str(row.get("sample_id") or ""),
            str(row.get("sample_dir") or ""),
        ),
    )


def build_frozen_split_manifest(dataset_root: Path, split_root: Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    split_root = Path(split_root)
    frozen_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in split_manifest_rows(split_root):
        sample_dir = Path(row.get("sample_dir") or "")
        artifact_paths, artifact_hashes, system_label, valid, problems = _snapshot_artifacts(sample_dir)
        sample_id = str(row.get("sample_id") or sample_dir.name)
        frozen_row: dict[str, Any] = {
            "sample_id": sample_id,
            "graph2mat_sample_id": sample_id,
            "deeph_sample_id": sample_id,
            "split": row.get("split"),
            "sample_dir": str(sample_dir),
            "system_label": system_label,
            "valid": valid,
            "validation_problems": problems,
            "artifact_paths": artifact_paths,
            "artifact_sha256": artifact_hashes,
        }
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
    split_hash = canonical_sha256(hash_rows)
    split_counts = {
        split: sum(1 for row in frozen_rows if row.get("split") == split)
        for split in ("train", "validation", "test")
    }
    return {
        "schema": FROZEN_SPLIT_SCHEMA,
        "artifact_contract_version": CONTRACT_NAME,
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "split_hash": split_hash,
        "split_counts": split_counts,
        "valid": not warnings and bool(frozen_rows),
        "warnings": warnings,
        "rows": frozen_rows,
    }


def _dataset_sample_rows_from_validation(artifact_validation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in artifact_validation.get("snapshots") or []:
        if not isinstance(snapshot, dict):
            continue
        artifacts = dict(snapshot.get("present_artifacts") or {})
        rows.append(
            {
                "sample_dir": snapshot.get("snapshot_dir"),
                "system_label": snapshot.get("system_label"),
                "valid": bool(snapshot.get("valid")),
                "repair_required": bool(snapshot.get("repair_required")),
                "missing_required": list(snapshot.get("missing_required") or []),
                "errors": list(snapshot.get("errors") or []),
                "warnings": list(snapshot.get("warnings") or []),
                "artifact_paths": artifacts,
                "artifact_sha256": _artifact_hashes(artifacts),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("sample_dir") or ""))


def _non_empty_mapping(payload: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and bool(value):
            return True
    return False


def _non_empty_text(payload: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


# A usable SIESTA version string must contain a dotted numeric version
# (e.g. "5.4.2-11-g4e9a46060"). Environment noise captured from stderr
# ("Authorization required, but no authorization protocol specified") has no
# such token, so it can never pass the provenance gate again.
_SIESTA_VERSION_TOKEN = re.compile(r"\d+\.\d+")
# Line emitted by `siesta --version` build info: "Version         : 5.4.2-...".
_SIESTA_BUILD_INFO_VERSION_LINE = re.compile(r"^\s*Version\s*:\s*(\S+)", re.MULTILINE)


def looks_like_siesta_version(text: Any) -> bool:
    """True when ``text`` plausibly names a SIESTA version (contains X.Y)."""
    return isinstance(text, str) and bool(_SIESTA_VERSION_TOKEN.search(text))


def extract_siesta_version_from_text(text: Any) -> str | None:
    """Extract a validated SIESTA version from probe output / build info.

    Prefers the build-info ``Version : <token>`` line; otherwise the first
    line that looks like a version. Returns None when nothing validates —
    callers must NOT fall back to arbitrary first lines (that is how X11
    noise got recorded as a version).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    match = _SIESTA_BUILD_INFO_VERSION_LINE.search(text)
    if match and looks_like_siesta_version(match.group(1)):
        return match.group(1)
    for line in text.splitlines():
        line = line.strip()
        if line and looks_like_siesta_version(line):
            return line
    return None


def _non_empty_text_or_sequence(payload: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, tuple)) and any(str(item).strip() for item in value):
            return True
    return False


def _existing_material_path(dataset_root: Path, material: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = material.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if not path.is_absolute():
            path = dataset_root / path
        if path.exists() and path.is_file():
            return True
    return False


def sanitized_environment_provenance(material: dict[str, Any]) -> dict[str, Any]:
    environment = material.get("environment")
    if not isinstance(environment, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in ENVIRONMENT_PROVENANCE_KEYS:
        value = environment.get(key)
        if value in (None, "", {}, []):
            continue
        sanitized[key] = value
    return sanitized


def _environment_provenance_present(material: dict[str, Any]) -> bool:
    environment = sanitized_environment_provenance(material)
    return _non_empty_text(environment, "python_version") and _non_empty_text(environment, "platform")


def fdf_block_lines(path: Path, block_name: str) -> list[str]:
    if not path.exists():
        return []
    lower_name = block_name.lower()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    inside = False
    output: list[str] = []
    for line in lines:
        clean = _strip_fdf_comment(line)
        if not clean:
            continue
        lower = clean.lower()
        if lower == f"%block {lower_name}":
            inside = True
            continue
        if inside and lower == f"%endblock {lower_name}":
            return output
        if inside:
            output.append(clean)
    return output


def kpoint_summary(path: Path) -> dict[str, Any]:
    rows = fdf_block_lines(path, "kgrid_Monkhorst_Pack")
    return {
        "kgrid_monkhorst_pack": rows,
        "present": bool(rows),
    }


def spin_summary(path: Path) -> dict[str, str]:
    return fdf_directives(path, keys=SPIN_FLAG_KEYS)


def provenance_status(
    dataset_root: Path,
    material: dict[str, Any],
    *,
    strict_paper_ready: bool = False,
) -> dict[str, Any]:
    run_fdf_path = dataset_root / "RUN.fdf"
    status: dict[str, Any] = {
        "basis_provenance": _non_empty_mapping(material, "basis_file_sha256", "basis_hashes"),
        "pseudopotential_provenance": _non_empty_mapping(
            material,
            "pseudopotential_sha256",
            "pseudopotential_hashes",
            "pseudopotential_sha256_by_species",
        ),
        "material_identity": _non_empty_text(material, "label", "material_label", "material_id"),
        "siesta_input_provenance": run_fdf_path.exists() or _non_empty_text(
            material,
            "fdf_sha256",
            "siesta_input_sha256",
        ),
        # A non-empty string is NOT enough: archived datasets carried X11
        # noise as siesta_version. The text must look like a real version.
        "siesta_version_provenance": looks_like_siesta_version(
            material.get("siesta_version")
        ) or _existing_material_path(dataset_root, material, "siesta_version_source_file"),
        "siesta_command_line_provenance": _non_empty_text_or_sequence(material, "siesta_command_line"),
        "siesta_environment_provenance": _environment_provenance_present(material),
        "siesta_execution_log_provenance": _existing_material_path(
            dataset_root,
            material,
            "siesta_stdout_path",
            "run_out_path",
        ),
    }
    required_keys = [
        "basis_provenance",
        "pseudopotential_provenance",
        "material_identity",
        "siesta_input_provenance",
    ]
    if strict_paper_ready:
        required_keys.extend(
            [
                "siesta_version_provenance",
                "siesta_command_line_provenance",
                "siesta_environment_provenance",
                "siesta_execution_log_provenance",
            ]
        )
    missing = [key for key in required_keys if not status.get(key)]
    return {
        **status,
        "strict_paper_ready": strict_paper_ready,
        "valid": not missing,
        "missing": missing,
    }


def build_benchmark_dataset_manifest(
    dataset_root: Path,
    *,
    artifact_validation: dict[str, Any],
    frozen_split_manifest: dict[str, Any] | None = None,
    material_provenance: dict[str, Any] | None = None,
    generation_mode: str = "clean_one_pass",
    strict_paper_ready_provenance: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    material = material_provenance or {}
    samples = _dataset_sample_rows_from_validation(artifact_validation)
    labels = sorted(
        {str(row.get("system_label")) for row in samples if row.get("system_label")}
    )
    system_label = labels[0] if len(labels) == 1 else None
    warnings = list(artifact_validation.get("warnings") or [])
    if len(labels) > 1:
        warnings.append(f"ambiguous dataset SystemLabel values: {labels}")

    run_fdf_path = dataset_root / "RUN.fdf"
    provenance = provenance_status(
        dataset_root,
        material,
        strict_paper_ready=strict_paper_ready_provenance,
    )
    for missing_key in provenance["missing"]:
        warnings.append(f"missing dataset-level {missing_key}")
    split_hash = (frozen_split_manifest or {}).get("split_hash")
    identity_payload = {
        "artifact_contract_version": CONTRACT_NAME,
        "generation_mode": generation_mode,
        "material_label": material.get("label"),
        "samples": [
            {
                "sample_dir": row.get("sample_dir"),
                "artifact_sha256": row.get("artifact_sha256"),
            }
            for row in samples
        ],
        "split_hash": split_hash,
    }
    benchmark_dataset_id = f"joint_graph2mat_deeph_{canonical_sha256(identity_payload)[:16]}"
    valid = (
        bool(artifact_validation.get("valid"))
        and not any(not row.get("valid") for row in samples)
        and bool(provenance["valid"])
    )
    if frozen_split_manifest is not None:
        valid = valid and bool(frozen_split_manifest.get("valid"))
    return {
        "schema": MANIFEST_SCHEMA,
        "benchmark_dataset_id": benchmark_dataset_id,
        "dataset_root": str(dataset_root),
        "artifact_contract_version": CONTRACT_NAME,
        "generation_mode": generation_mode,
        "validation_status": "valid" if valid else "invalid",
        "benchmark_ready": valid,
        "warnings": warnings,
        "material_label": material.get("label"),
        "material_source": material,
        "system_label": system_label,
        "siesta_input_path": str(run_fdf_path) if run_fdf_path.exists() else "",
        "siesta_input_sha256": file_sha256(run_fdf_path) if run_fdf_path.exists() else "",
        "siesta_flags": fdf_directives(run_fdf_path),
        "siesta_version": material.get("siesta_version", ""),
        "siesta_version_source_file": material.get("siesta_version_source_file", ""),
        "siesta_executable": material.get("siesta_executable", ""),
        "siesta_command_line": material.get("siesta_command_line", ""),
        "siesta_stdout_path": material.get("siesta_stdout_path") or material.get("run_out_path") or "",
        "siesta_returncode": material.get("siesta_returncode"),
        "siesta_build_info": material.get("siesta_build_info", ""),
        "kpoint_summary": kpoint_summary(run_fdf_path),
        "spin_summary": spin_summary(run_fdf_path),
        "environment": sanitized_environment_provenance(material),
        "graph2mat_commit": material.get("graph2mat_commit", ""),
        "deeph_pack_commit": material.get("deeph_pack_commit", ""),
        "basis_hashes": material.get("basis_file_sha256") or {},
        "pseudopotential_hashes": material.get("pseudopotential_sha256") or {},
        "provenance_status": provenance,
        "artifact_validation": artifact_validation,
        "samples": samples,
        "frozen_split_manifest": {
            "path": str(dataset_root / "frozen_split_manifest.json"),
            "split_hash": split_hash,
            "split_counts": (frozen_split_manifest or {}).get("split_counts", {}),
            "valid": (frozen_split_manifest or {}).get("valid"),
        },
    }


def write_benchmark_manifests(
    *,
    dataset_root: Path,
    split_root: Path,
    generation_mode: str = "clean_one_pass",
    artifact_validation_path: Path | None = None,
    material_provenance_path: Path | None = None,
    strict_paper_ready_provenance: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_root = Path(dataset_root)
    artifact_validation = read_json(artifact_validation_path or dataset_root / "artifact_validation.json")
    material_provenance = read_json(material_provenance_path or dataset_root / "material_provenance.json")
    frozen_split = build_frozen_split_manifest(dataset_root, split_root)
    dataset_manifest = build_benchmark_dataset_manifest(
        dataset_root,
        artifact_validation=artifact_validation,
        frozen_split_manifest=frozen_split,
        material_provenance=material_provenance,
        generation_mode=generation_mode,
        strict_paper_ready_provenance=strict_paper_ready_provenance,
    )
    write_json(dataset_root / "frozen_split_manifest.json", frozen_split)
    write_json(dataset_root / "benchmark_dataset_manifest.json", dataset_manifest)
    if not dataset_manifest["benchmark_ready"]:
        raise RuntimeError(
            "Benchmark dataset manifest is not valid; refusing to freeze dataset for training. "
            f"See {dataset_root / 'benchmark_dataset_manifest.json'}"
        )
    return dataset_manifest, frozen_split
