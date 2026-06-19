#!/usr/bin/env python3
"""Validate derivative workflow artifact layout and metadata consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PREDICTION_MANIFESTS = {
    "graph2mat": "derivative_graph2mat_prediction_manifest.json",
    "deeph": "derivative_deeph_prediction_manifest.json",
}
MODEL_SUBROOTS = {
    "graph2mat": "graph2mat_derivative_result",
    "deeph": "deeph_derivative_result",
}
METRICS_ROOT_NAME = "derivative_metrics"
WORKFLOW_MANIFEST_NAME = "derivative_workflow_manifest.json"
STENCIL_MANIFEST_NAME = "derivative_stencil_manifest.json"
REFERENCE_MANIFEST_NAME = "derivative_siesta_reference_manifest.json"
GATE_REPORT_NAME = "derivative_gate_report.json"
PREDICTION_FILENAME = "ML_prediction.HSX"


class DerivativeArtifactValidationError(RuntimeError):
    """Raised when derivative artifact validation cannot complete."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DerivativeArtifactValidationError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DerivativeArtifactValidationError(f"Malformed JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DerivativeArtifactValidationError(f"JSON payload must be an object: {path}")
    return payload


def read_json_or_error(path: Path, *, errors: list[str]) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except DerivativeArtifactValidationError as exc:
        errors.append(str(exc))
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def scope_has_direct_artifacts(root: Path) -> bool:
    return any(
        (root / name).exists()
        for name in (
            "structures",
            STENCIL_MANIFEST_NAME,
            "siesta_hamiltonians",
            "predicted_hamiltonians",
            METRICS_ROOT_NAME,
        )
    )


def discover_validation_scopes(root: Path, *, model: str) -> list[Path]:
    scopes: list[Path] = []
    if scope_has_direct_artifacts(root):
        scopes.append(root)
    target_models = ("graph2mat", "deeph") if model == "auto" else (model,)
    for target_model in target_models:
        subroot = root / MODEL_SUBROOTS[target_model]
        if subroot.exists():
            scopes.append(subroot)
    deduped: list[Path] = []
    seen: set[str] = set()
    for scope in scopes:
        key = str(scope.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(scope)
    return deduped


def scope_model_for(root: Path, *, model: str) -> str:
    if root.name == MODEL_SUBROOTS["graph2mat"]:
        return "graph2mat"
    if root.name == MODEL_SUBROOTS["deeph"]:
        return "deeph"
    return model


def path_mentions_model(path: Path, model: str) -> bool:
    if model == "auto":
        return True
    haystack = " ".join(path.parts + (path.name,))
    return model in haystack.lower()


def workflow_manifest_for(root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for candidate in (root, *root.parents):
        path = candidate / WORKFLOW_MANIFEST_NAME
        if path.exists():
            return path, read_json(path)
    return None, None


def required_fields_present(payload: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if not as_text(payload.get(field))]


def load_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise DerivativeArtifactValidationError(f"Missing sample metadata: {metadata_path}")
    return read_json(metadata_path)


def record_checked(*, checked: list[str], checked_paths: list[str], label: str, path: Path) -> None:
    checked.append(f"{label}:{path}")
    checked_paths.append(str(path))


def validate_stencil_manifest(root: Path, *, checked: list[str], checked_paths: list[str], warnings: list[str], errors: list[str]) -> None:
    manifest_path = root / STENCIL_MANIFEST_NAME
    structures_root = root / "structures"
    if not structures_root.exists():
        return
    record_checked(checked=checked, checked_paths=checked_paths, label="stencil_manifest", path=manifest_path)
    if not manifest_path.exists():
        errors.append(f"Missing derivative stencil manifest for structures tree: {manifest_path}")
        return

    manifest = read_json(manifest_path)
    samples = manifest.get("samples")
    stencils = manifest.get("stencils")
    if not isinstance(samples, list):
        errors.append(f"Stencil manifest samples list is missing or invalid: {manifest_path}")
        return
    if not isinstance(stencils, list):
        errors.append(f"Stencil manifest stencils list is missing or invalid: {manifest_path}")
        return

    sample_records: dict[str, dict[str, Any]] = {}
    for raw_sample in samples:
        if not isinstance(raw_sample, dict):
            errors.append(f"Stencil manifest sample entry is not an object: {manifest_path}")
            continue
        sample_id = as_text(raw_sample.get("sample_id"))
        if not sample_id:
            errors.append(f"Stencil manifest sample is missing sample_id: {manifest_path}")
            continue
        sample_dir = structures_root / sample_id
        if not sample_dir.exists():
            errors.append(f"Missing sample directory: {sample_dir}")
            continue
        if not file_exists(sample_dir / "RUN.fdf"):
            errors.append(f"Missing RUN.fdf: {sample_dir / 'RUN.fdf'}")
        if not file_exists(sample_dir / "metadata.json"):
            errors.append(f"Missing metadata.json: {sample_dir / 'metadata.json'}")
            continue
        try:
            metadata = load_sample_metadata(sample_dir)
        except DerivativeArtifactValidationError as exc:
            errors.append(str(exc))
            continue
        required = ("base_sample_id", "atom_index_zero_based", "axis", "delta_ang", "finite_difference_method")
        missing = required_fields_present(metadata, required)
        if missing:
            errors.append(f"Sample metadata is missing required fields for {sample_id}: {', '.join(missing)}")
        sample_records[sample_id] = {
            "sample_dir": sample_dir,
            "metadata": metadata,
            "sample": raw_sample,
        }

    if not sample_records:
        errors.append(f"No valid sample records were found in {manifest_path}")
        return

    for raw_stencil in stencils:
        if not isinstance(raw_stencil, dict):
            errors.append(f"Stencil manifest stencil entry is not an object: {manifest_path}")
            continue
        method = as_text(raw_stencil.get("finite_difference_method")) or as_text(manifest.get("finite_difference_method"))
        base_id = as_text(raw_stencil.get("base_sample_id"))
        plus_id = as_text(raw_stencil.get("plus_sample_id"))
        minus_id = as_text(raw_stencil.get("minus_sample_id"))
        family_ids = [sample_id for sample_id in (base_id, plus_id, minus_id) if sample_id]
        if not method:
            errors.append(f"Stencil record is missing finite_difference_method: {manifest_path}")
            continue
        if method == "central" and (not base_id or not plus_id or not minus_id):
            errors.append(f"Central stencil family is missing R0/R+/R- members: {family_ids}")
            continue
        if base_id and base_id not in sample_records:
            errors.append(f"Stencil references missing base sample: {base_id}")
            continue
        for sample_id in family_ids:
            if sample_id not in sample_records:
                errors.append(f"Stencil references missing sample: {sample_id}")
                continue
        if any(sample_id not in sample_records for sample_id in family_ids):
            continue
        base_meta = sample_records[base_id]["metadata"] if base_id else {}
        family_metas = [sample_records[sample_id]["metadata"] for sample_id in family_ids]
        family_fields = ("base_sample_id", "atom_index_zero_based", "axis", "finite_difference_method", "split")
        for field in family_fields:
            values = {as_text(meta.get(field)) for meta in family_metas if as_text(meta.get(field))}
            if len(values) > 1:
                errors.append(f"Stencil family {family_ids} has inconsistent {field}: {sorted(values)}")
        displaced_metas = [sample_records[sample_id]["metadata"] for sample_id in (plus_id, minus_id) if sample_id]
        if displaced_metas:
            displaced_fields = ("atom_index_zero_based", "axis", "delta_ang", "finite_difference_method", "split")
            for field in displaced_fields:
                values = {as_text(meta.get(field)) for meta in displaced_metas if as_text(meta.get(field))}
                if len(values) > 1:
                    errors.append(f"Stencil family {family_ids} has inconsistent displaced-member {field}: {sorted(values)}")
        if base_id:
            expected_base = as_text(base_meta.get("sample_id")) or base_id
            if expected_base and any(as_text(meta.get("base_sample_id")) not in {expected_base, base_id} for meta in family_metas):
                errors.append(f"Stencil family {family_ids} has inconsistent base_sample_id metadata")


def validate_reference_manifests(root: Path, *, checked: list[str], checked_paths: list[str], warnings: list[str], errors: list[str]) -> None:
    manifest_path = root / "siesta_hamiltonians" / REFERENCE_MANIFEST_NAME
    if not manifest_path.exists():
        return
    record_checked(checked=checked, checked_paths=checked_paths, label="reference_manifest", path=manifest_path)
    manifests = [manifest_path]
    for manifest_path in manifests:
        manifest = read_json_or_error(manifest_path, errors=errors)
        if manifest is None:
            continue
        rows = manifest.get("rows")
        if not isinstance(rows, list):
            errors.append(f"Reference manifest has no rows list: {manifest_path}")
            continue
        for row in rows:
            if not isinstance(row, dict):
                errors.append(f"Reference manifest row is not an object: {manifest_path}")
                continue
            status = as_text(row.get("status"))
            if status not in {"ok", "staged", "skipped_existing"}:
                continue
            reference_dir = Path(as_text(row.get("reference_dir")))
            if not reference_dir.exists():
                errors.append(f"Missing SIESTA reference directory: {reference_dir}")
                continue
            reference_matrix = as_text(row.get("reference_matrix"))
            if not reference_matrix:
                errors.append(f"Reference manifest row is missing reference_matrix: {manifest_path}")
                continue
            reference_matrix_path = Path(reference_matrix)
            if not file_exists(reference_matrix_path):
                errors.append(f"Missing SIESTA reference matrix: {reference_matrix_path}")
            if reference_matrix_path.parent != reference_dir and not reference_dir.exists():
                warnings.append(f"Reference matrix is outside its reference_dir: {reference_matrix_path}")


def validate_prediction_manifests(root: Path, *, model: str, checked: list[str], checked_paths: list[str], warnings: list[str], errors: list[str]) -> None:
    target_models = ("graph2mat", "deeph") if model == "auto" else (model,)
    for target_model in target_models:
        manifest_path = root / "predicted_hamiltonians" / PREDICTION_MANIFESTS[target_model]
        if not manifest_path.exists():
            continue
        record_checked(checked=checked, checked_paths=checked_paths, label="prediction_manifest", path=manifest_path)
        manifest = read_json_or_error(manifest_path, errors=errors)
        if manifest is None:
            continue
        rows = manifest.get("rows")
        if not isinstance(rows, list):
            errors.append(f"Prediction manifest has no rows list: {manifest_path}")
            continue
        for row in rows:
            if not isinstance(row, dict):
                errors.append(f"Prediction manifest row is not an object: {manifest_path}")
                continue
            status = as_text(row.get("status"))
            if status not in {"predicted", "staged", "skipped_existing"}:
                continue
            prediction_dir = Path(as_text(row.get("prediction_dir")))
            if not prediction_dir.exists():
                errors.append(f"Missing prediction directory: {prediction_dir}")
                continue
            prediction_path = as_text(row.get("prediction_path")) or str(prediction_dir / PREDICTION_FILENAME)
            prediction_file = Path(prediction_path)
            if not file_exists(prediction_file):
                errors.append(f"Missing prediction file: {prediction_file}")


def validate_metrics_manifests(root: Path, *, model: str, checked: list[str], checked_paths: list[str], warnings: list[str], errors: list[str]) -> None:
    metrics_root = root / METRICS_ROOT_NAME
    if not metrics_root.exists():
        return
    target_models = ("graph2mat", "deeph") if model == "auto" else (model,)
    for target_model in target_models:
        manifest_path = metrics_root / target_model / "manifest.json"
        if not manifest_path.exists():
            continue
        record_checked(checked=checked, checked_paths=checked_paths, label="metrics_manifest", path=manifest_path)
        manifest = read_json_or_error(manifest_path, errors=errors)
        if manifest is None:
            continue
        delta_stability_path = manifest_path.parent / "derivative_delta_stability.json"
        if not file_exists(delta_stability_path):
            errors.append(f"Missing derivative_delta_stability.json: {delta_stability_path}")


def validate_gate_reports(root: Path, *, model: str, checked: list[str], checked_paths: list[str], warnings: list[str], errors: list[str]) -> None:
    metrics_root = root / METRICS_ROOT_NAME
    if not metrics_root.exists():
        return
    target_models = ("graph2mat", "deeph") if model == "auto" else (model,)
    for target_model in target_models:
        for report_path in sorted({path for path in (metrics_root / target_model).rglob(GATE_REPORT_NAME) if path.is_file()}):
            record_checked(checked=checked, checked_paths=checked_paths, label="gate_report", path=report_path)
            read_json_or_error(report_path, errors=errors)


def stage_statuses(workflow_manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    stages = workflow_manifest.get("stages") if isinstance(workflow_manifest, dict) else None
    if not isinstance(stages, dict):
        return {}
    return {str(stage): record for stage, record in stages.items() if isinstance(record, dict)}


def validate_completed_stage_artifacts(
    root: Path,
    *,
    workflow_manifest: dict[str, Any] | None,
    model: str,
    checked: list[str],
    warnings: list[str],
    errors: list[str],
) -> None:
    stages = stage_statuses(workflow_manifest)
    if not stages:
        return
    scope_model = scope_model_for(root, model=model)
    target_models = ("graph2mat", "deeph") if scope_model == "auto" else (scope_model,)
    if stages.get("build_derivative_stencils", {}).get("status") == "completed" and not (root / STENCIL_MANIFEST_NAME).exists():
        errors.append(f"Workflow manifest says build_derivative_stencils completed, but no {STENCIL_MANIFEST_NAME} was found under {root}")
    if stages.get("run_derivative_siesta_reference", {}).get("status") == "completed" and not (root / "siesta_hamiltonians" / REFERENCE_MANIFEST_NAME).exists():
        errors.append(f"Workflow manifest says run_derivative_siesta_reference completed, but no {REFERENCE_MANIFEST_NAME} was found under {root}")
    if "graph2mat" in target_models and stages.get("predict_derivative_graph2mat", {}).get("status") == "completed" and not (root / "predicted_hamiltonians" / PREDICTION_MANIFESTS["graph2mat"]).exists():
        errors.append(f"Workflow manifest says predict_derivative_graph2mat completed, but no {PREDICTION_MANIFESTS['graph2mat']} was found under {root}")
    if "deeph" in target_models and stages.get("predict_derivative_deeph", {}).get("status") == "completed" and not (root / "predicted_hamiltonians" / PREDICTION_MANIFESTS["deeph"]).exists():
        errors.append(f"Workflow manifest says predict_derivative_deeph completed, but no {PREDICTION_MANIFESTS['deeph']} was found under {root}")
    if "graph2mat" in target_models and stages.get("derivative_metrics_graph2mat", {}).get("status") == "completed" and not (root / METRICS_ROOT_NAME / "graph2mat" / "manifest.json").exists():
        errors.append(f"Workflow manifest says derivative_metrics_graph2mat completed, but no {METRICS_ROOT_NAME}/manifest.json was found under {root}")
    if "deeph" in target_models and stages.get("derivative_metrics_deeph", {}).get("status") == "completed" and not (root / METRICS_ROOT_NAME / "deeph" / "manifest.json").exists():
        errors.append(f"Workflow manifest says derivative_metrics_deeph completed, but no {METRICS_ROOT_NAME}/manifest.json was found under {root}")
    gate_reports = []
    for target_model in target_models:
        gate_reports.extend(path for path in (root / METRICS_ROOT_NAME / target_model).rglob(GATE_REPORT_NAME) if path.is_file())
    if stages.get("derivative_gate_check", {}).get("status") == "completed" and not gate_reports:
        errors.append(f"Workflow manifest says derivative_gate_check completed, but no {GATE_REPORT_NAME} was found under {root}")


def validate_derivative_workflow_artifacts(
    derivative_result_root: Path,
    *,
    output_json: Path | None = None,
    fail_on_warning: bool = False,
    model: str = "auto",
) -> dict[str, Any]:
    root = Path(derivative_result_root)
    if model not in {"auto", "graph2mat", "deeph"}:
        raise DerivativeArtifactValidationError("--model must be one of: auto, graph2mat, deeph.")
    if not root.exists():
        raise DerivativeArtifactValidationError(f"Derivative result root does not exist: {root}")

    checked: list[str] = [f"root_exists:{root}"]
    checked_roots: list[str] = []
    checked_paths: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    workflow_manifest_path = None
    workflow_manifest = None
    try:
        workflow_manifest_path, workflow_manifest = workflow_manifest_for(root)
    except DerivativeArtifactValidationError as exc:
        errors.append(str(exc))
    else:
        if workflow_manifest_path is not None:
            checked.append(f"workflow_manifest:{workflow_manifest_path}")

    scopes = discover_validation_scopes(root, model=model)
    if not scopes:
        errors.append(
            "No derivative artifact scope found under "
            f"{root}; expected direct artifacts (structures/, derivative_stencil_manifest.json, "
            "siesta_hamiltonians/, predicted_hamiltonians/, derivative_metrics/) or model subroots "
            "graph2mat_derivative_result/ and deeph_derivative_result/."
        )
    for scope_root in scopes:
        checked_roots.append(str(scope_root))
        validate_stencil_manifest(scope_root, checked=checked, checked_paths=checked_paths, warnings=warnings, errors=errors)
        validate_reference_manifests(scope_root, checked=checked, checked_paths=checked_paths, warnings=warnings, errors=errors)
        validate_prediction_manifests(scope_root, model=model, checked=checked, checked_paths=checked_paths, warnings=warnings, errors=errors)
        validate_metrics_manifests(scope_root, model=model, checked=checked, checked_paths=checked_paths, warnings=warnings, errors=errors)
        validate_gate_reports(scope_root, model=model, checked=checked, checked_paths=checked_paths, warnings=warnings, errors=errors)
        validate_completed_stage_artifacts(
            scope_root,
            workflow_manifest=workflow_manifest,
            model=model,
            checked=checked,
            warnings=warnings,
            errors=errors,
        )

    status = "failed" if errors or (fail_on_warning and warnings) else "warning" if warnings else "ok"
    summary = {
        "status": status,
        "root": str(root),
        "checked_roots": checked_roots,
        "checked_paths": checked_paths,
        "checked": checked,
        "warnings": warnings,
        "errors": errors,
    }
    if output_json is not None:
        write_json(Path(output_json), summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derivative_result_root", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--model", choices=["auto", "graph2mat", "deeph"], default="auto")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        summary = validate_derivative_workflow_artifacts(
            args.derivative_result_root,
            output_json=args.output_json,
            fail_on_warning=bool(args.fail_on_warning),
            model=args.model,
        )
    except DerivativeArtifactValidationError as exc:
        print(f"[DERIVATIVE-ARTIFACTS][ERROR] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 2 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
