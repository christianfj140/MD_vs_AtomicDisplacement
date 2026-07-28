"""Fail-closed contract for material-specific electronic convergence evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "electronic_convergence_evidence_v1"
REQUIRED_STUDIES = ("mesh_cutoff", "kpoint_density", "scf_tolerance")


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_electronic_convergence(
    payload: dict[str, Any],
    *,
    material_label: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if material_label and payload.get("material_label") != material_label:
        errors.append("material_label does not match the dataset")
    if payload.get("predeclared") is not True:
        errors.append("thresholds/protocol were not predeclared")
    observable = payload.get("observable")
    if not isinstance(observable, dict) or not observable.get("name") or not observable.get("unit"):
        errors.append("observable name and unit are required")
    tolerance = payload.get("tolerance")
    if (
        not isinstance(tolerance, dict)
        or not isinstance(tolerance.get("value"), (int, float))
        or float(tolerance["value"]) <= 0
        or not tolerance.get("unit")
        or not tolerance.get("justification")
    ):
        errors.append("positive, justified tolerance with unit is required")
    studies = payload.get("studies")
    if not isinstance(studies, dict):
        studies = {}
        errors.append("studies must be an object")
    required = list(REQUIRED_STUDIES)
    if payload.get("basis_convergence_required") is True:
        required.append("basis")
    for name in required:
        study = studies.get(name)
        if not isinstance(study, dict):
            errors.append(f"missing convergence study: {name}")
            continue
        points = study.get("points")
        if not isinstance(points, list) or len(points) < 3:
            errors.append(f"{name} requires at least three points")
        elif any(
            not isinstance(point, dict)
            or point.get("parameter") in (None, "")
            or not isinstance(point.get("observable"), (int, float))
            for point in points
        ):
            errors.append(f"{name} points require parameter and numeric observable")
        if study.get("converged") is not True:
            errors.append(f"{name} is not converged")
    if payload.get("converged") is not True:
        errors.append("overall electronic convergence is not established")
    return errors


def convergence_status(dataset_root: Path, *, material_label: str | None = None) -> dict[str, Any]:
    path = Path(dataset_root) / "electronic_convergence.json"
    if not path.is_file():
        return {
            "status": "exact_configuration_only",
            "pooling_allowed": False,
            "path": str(path),
            "errors": ["electronic_convergence.json is missing"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "pooling_allowed": False,
            "path": str(path),
            "errors": [f"invalid electronic_convergence.json: {exc}"],
        }
    if not isinstance(payload, dict):
        errors = ["electronic_convergence.json root must be an object"]
    else:
        errors = validate_electronic_convergence(payload, material_label=material_label)
    return {
        "status": "converged" if not errors else "invalid",
        "pooling_allowed": not errors,
        "path": str(path),
        "evidence_sha256": canonical_sha256(payload) if isinstance(payload, dict) else "",
        "equivalence_class_hash": (
            str(payload.get("equivalence_class_hash") or "")
            if isinstance(payload, dict)
            else ""
        ),
        "errors": errors,
    }


def pooling_errors(datasets: list[tuple[Path, str | None]]) -> list[str]:
    """Require validated, matching convergence evidence only for cross-dataset pooling."""

    if len(datasets) <= 1:
        return []
    statuses = [
        (root, convergence_status(root, material_label=label))
        for root, label in datasets
    ]
    errors = [
        f"{root}: {error}"
        for root, status in statuses
        for error in status["errors"]
    ]
    classes = {
        status["equivalence_class_hash"]
        for _, status in statuses
        if status["pooling_allowed"]
    }
    if len(classes) != 1 or "" in classes:
        errors.append(
            "cross-dataset pooling requires one non-empty electronic equivalence_class_hash"
        )
    return errors
