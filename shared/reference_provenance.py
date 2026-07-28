"""Canonical positive provenance for one SIESTA reference calculation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from siesta_output_status import parse_siesta_output


SCHEMA = "positive_siesta_reference_provenance_v3"
CALCULATION_ID_FIELDS = (
    "frozen_sample_id",
    "reference_format",
    "reference_sha256",
    "run_fdf_sha256",
    "run_out_sha256",
    "geometry_cell_species_sha256",
    "system_label",
    "orb_indx_sha256",
    "basis_hashes",
    "pseudopotential_hashes",
    "siesta_version",
    "siesta_command",
    "frozen_split_hash",
    "split",
)


def file_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
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
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def system_label_from_fdf(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    for line in lines:
        parts = line.split("#", 1)[0].split()
        if len(parts) >= 2 and parts[0].lower() == "systemlabel":
            return parts[1]
    return ""


def geometry_cell_species_sha256(path: Path) -> str:
    """Hash the FDF cell, species and geometry semantics, excluding SCF controls."""

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    scalars = {
        "numberofatoms",
        "numberofspecies",
        "latticeconstant",
        "atomiccoordinatesformat",
    }
    blocks = {
        "chemicalspecieslabel",
        "latticevectors",
        "atomiccoordinatesandatomicspecies",
    }
    selected: list[str] = []
    active_block = ""
    for raw in lines:
        clean = " ".join(raw.split("#", 1)[0].split())
        if not clean:
            continue
        parts = clean.split()
        lower = parts[0].lower()
        if lower == "%block" and len(parts) > 1:
            active_block = parts[1].lower()
            if active_block in blocks:
                selected.append(f"%block {active_block}")
            continue
        if lower == "%endblock":
            if active_block in blocks:
                selected.append(f"%endblock {active_block}")
            active_block = ""
            continue
        if active_block in blocks or lower in scalars:
            selected.append(clean.lower())
    return canonical_sha256(selected) if selected else ""


def calculation_id(payload: dict[str, Any]) -> str:
    return canonical_sha256({key: payload.get(key) for key in CALCULATION_ID_FIELDS})


def build_positive_reference_provenance(
    sample_dir: Path,
    reference_path: Path,
    *,
    frozen_sample_id: str,
    split: str,
    frozen_split_hash: str,
    basis_hashes: dict[str, str],
    pseudopotential_hashes: dict[str, str],
    siesta_version: str,
    siesta_command: str | list[str],
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    reference_path = Path(reference_path)
    run_fdf = sample_dir / "RUN.fdf"
    run_out = next(
        (path for path in (sample_dir / "RUN.out", sample_dir / "siesta.out") if path.is_file()),
        sample_dir / "RUN.out",
    )
    label = system_label_from_fdf(run_fdf)
    orb_indx = sample_dir / f"{label}.ORB_INDX"
    status = parse_siesta_output(run_out, run_fdf)
    timestamp = (
        datetime.fromtimestamp(run_out.stat().st_mtime, tz=timezone.utc).isoformat()
        if run_out.is_file()
        else ""
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "positive_siesta_provenance_valid",
        "frozen_sample_id": str(frozen_sample_id),
        "calculation_id": "",
        "snapshot_dir": str(sample_dir),
        "reference_path": str(reference_path),
        "reference_format": reference_path.suffix.removeprefix(".").upper(),
        "reference_sha256": file_sha256(reference_path),
        "run_fdf_sha256": file_sha256(run_fdf),
        "run_out_sha256": file_sha256(run_out),
        "geometry_cell_species_sha256": geometry_cell_species_sha256(run_fdf),
        "system_label": label,
        "orb_indx_sha256": file_sha256(orb_indx),
        "basis_hashes": dict(sorted(basis_hashes.items())),
        "pseudopotential_hashes": dict(sorted(pseudopotential_hashes.items())),
        "siesta_version": str(siesta_version),
        "siesta_command": siesta_command,
        "canonical_scf_status": {
            "valid": status.get("valid") is True,
            "parser_status": status.get("parser_status"),
            "job_completed": status.get("job_completed") is True,
            "scf_converged": status.get("scf_converged") is True,
        },
        "timestamp": timestamp,
        "frozen_split_hash": str(frozen_split_hash),
        "split": str(split),
    }
    payload["calculation_id"] = calculation_id(payload)
    required = [
        payload.get("frozen_sample_id"),
        payload.get("reference_sha256"),
        payload.get("run_fdf_sha256"),
        payload.get("run_out_sha256"),
        payload.get("geometry_cell_species_sha256"),
        payload.get("system_label"),
        payload.get("orb_indx_sha256"),
        payload.get("basis_hashes"),
        payload.get("pseudopotential_hashes"),
        payload.get("siesta_version"),
        payload.get("siesta_command"),
        payload.get("timestamp"),
        payload.get("frozen_split_hash"),
        payload.get("split"),
    ]
    if not all(required) or status.get("valid") is not True:
        payload["status"] = "invalid"
    return payload


def validate_positive_reference_provenance(
    payload: dict[str, Any],
    *,
    sample_dir: Path,
    reference_path: Path,
) -> list[str]:
    sample_dir = Path(sample_dir)
    reference_path = Path(reference_path)
    label = system_label_from_fdf(sample_dir / "RUN.fdf")
    actual = {
        "reference_format": reference_path.suffix.removeprefix(".").upper(),
        "reference_sha256": file_sha256(reference_path),
        "run_fdf_sha256": file_sha256(sample_dir / "RUN.fdf"),
        "run_out_sha256": file_sha256(
            next(
                (path for path in (sample_dir / "RUN.out", sample_dir / "siesta.out") if path.is_file()),
                sample_dir / "RUN.out",
            )
        ),
        "geometry_cell_species_sha256": geometry_cell_species_sha256(sample_dir / "RUN.fdf"),
        "system_label": label,
        "orb_indx_sha256": file_sha256(sample_dir / f"{label}.ORB_INDX"),
    }
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("legacy_or_missing_reference_provenance_schema")
    if payload.get("status") != "positive_siesta_provenance_valid":
        errors.append("reference_provenance_status_invalid")
    for key, value in actual.items():
        if not value or payload.get(key) != value:
            errors.append(f"{key}_mismatch")
    for key in (
        "frozen_sample_id",
        "calculation_id",
        "basis_hashes",
        "pseudopotential_hashes",
        "siesta_version",
        "siesta_command",
        "timestamp",
        "frozen_split_hash",
        "split",
    ):
        if payload.get(key) in (None, "", {}, []):
            errors.append(f"missing_{key}")
    scf = payload.get("canonical_scf_status")
    if not isinstance(scf, dict) or scf.get("valid") is not True or scf.get("parser_status") != "valid":
        errors.append("canonical_scf_status_invalid")
    if payload.get("calculation_id") != calculation_id(payload):
        errors.append("calculation_id_mismatch")
    return errors
