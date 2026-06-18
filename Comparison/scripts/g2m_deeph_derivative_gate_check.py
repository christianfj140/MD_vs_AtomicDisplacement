#!/usr/bin/env python3
"""Fail-closed scientific gate checker for Hamiltonian derivative comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "graph2mat_deeph_derivative_gate_report_v1"
EXPECTED_REFERENCE_DEFINITION = "siesta_hamiltonian_finite_difference"
EXPECTED_DERIVATIVE_UNITS = "eV/Ang"
DEFAULT_HERMITICITY_THRESHOLD = 1e-8
DEFAULT_SUPPORT_DISCONTINUITY_THRESHOLD = 1e-12
VALID_STATUSES = ("internal_diagnostic", "technical_presentation", "paper_level_candidate", "blocked")
STATUS_RANK = {status: index for index, status in enumerate(VALID_STATUSES)}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON payload must be an object: {path}")
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_root(root: Path) -> Path:
    root = Path(root)
    return root if root.name == "derivative_metrics" else root / "derivative_metrics"


def infer_model(root: Path) -> str:
    parts = [root.name.lower(), root.parent.name.lower(), root.parent.parent.name.lower() if len(root.parents) > 1 else ""]
    if any("graph2mat" in part or part == "g2m" for part in parts):
        return "Graph2Mat"
    if any("deeph" in part for part in parts):
        return "DeepH"
    return root.parent.name or root.name


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def unique_nonempty(values: list[Any]) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def path_map(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "matrix_metrics": root / "derivative_matrix_metrics.csv",
        "stencil_status": root / "stencil_status.csv",
        "hermiticity": root / "derivative_hermiticity.csv",
        "delta_stability": root / "derivative_delta_stability.json",
    }


def load_dataset(root: Path) -> dict[str, Any]:
    root = normalize_root(root)
    paths = path_map(root)
    manifest = read_json(paths["manifest"])
    delta_stability = read_json(paths["delta_stability"]) if paths["delta_stability"].exists() else {}
    return {
        "root": root,
        "model": infer_model(root),
        "paths": paths,
        "manifest": manifest,
        "delta_stability": delta_stability,
        "matrix_rows": read_csv_rows(paths["matrix_metrics"]),
        "stencil_rows": read_csv_rows(paths["stencil_status"]),
        "hermiticity_rows": read_csv_rows(paths["hermiticity"]),
    }


def gate_row(
    gate_id: str,
    *,
    severity: str,
    status: str,
    message: str,
    model: str | None = None,
    blocks_status: str | None = None,
    claim_scope: str = "all",
    evidence_paths: list[Path | str] | None = None,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "model": model,
        "severity": severity,
        "status": status,
        "message": message,
        "blocks_status": blocks_status,
        "claim_scope": claim_scope,
        "evidence_paths": [str(path) for path in (evidence_paths or []) if str(path)],
    }


def warning_row(
    warning_id: str,
    *,
    message: str,
    model: str | None = None,
    severity: str = "warning",
    evidence_paths: list[Path | str] | None = None,
) -> dict[str, Any]:
    return gate_row(
        warning_id,
        model=model,
        severity=severity,
        status="warning",
        message=message,
        evidence_paths=evidence_paths,
    )


def matches_text(payload: Any, terms: tuple[str, ...]) -> bool:
    if isinstance(payload, dict):
        return any(matches_text(value, terms) or matches_text(key, terms) for key, value in payload.items())
    if isinstance(payload, list):
        return any(matches_text(item, terms) for item in payload)
    text = str(payload or "").lower()
    return any(term in text for term in terms)


def central_stencil_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        row for row in dataset["stencil_rows"]
        if str(row.get("finite_difference_method") or dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central"
    ]
    if rows:
        return rows
    if str(dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central":
        return dataset["stencil_rows"]
    return []


def central_metric_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        row for row in dataset["matrix_rows"]
        if str(row.get("finite_difference_method") or dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central"
    ]
    if rows:
        return rows
    if str(dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central":
        return dataset["matrix_rows"]
    return []


def max_hermiticity_defect(dataset: dict[str, Any]) -> float:
    values: list[float] = []
    for row in dataset["hermiticity_rows"]:
        for key in ("dH_ref_hermiticity_defect", "dH_pred_hermiticity_defect", "dH_hermiticity_error_delta"):
            value = number(row.get(key))
            if value is not None:
                values.append(value)
    for row in dataset["matrix_rows"]:
        for key in ("dh_hermiticity_ref", "dh_hermiticity_pred", "dh_hermiticity_error_delta"):
            value = number(row.get(key))
            if value is not None:
                values.append(value)
    return max(values) if values else 0.0


def support_discontinuity_detected(dataset: dict[str, Any], threshold: float) -> bool:
    for row in dataset["matrix_rows"]:
        if truthy(row.get("dh_support_changed")):
            return True
        if (number(row.get("dh_false_zero_rate")) or 0.0) > threshold:
            return True
        if (number(row.get("dh_false_nonzero_rate")) or 0.0) > threshold:
            return True
        if truthy(row.get("reference_plus_minus_support_changed")) or truthy(row.get("predicted_plus_minus_support_changed")):
            return True
    return False


def explicit_issue(dataset: dict[str, Any], terms: tuple[str, ...]) -> bool:
    manifest = dataset["manifest"]
    if matches_text(manifest.get("fatal_errors"), terms):
        return True
    if matches_text(manifest.get("warnings"), terms):
        return True
    for row in dataset["stencil_rows"]:
        if matches_text(row.get("issue_codes"), terms) or matches_text(row.get("issue_messages"), terms):
            return True
    return False


def dataset_paper_evidence(dataset: dict[str, Any]) -> dict[str, bool]:
    manifest = dataset["manifest"]
    matrix_rows = dataset["matrix_rows"]
    delta_stability = dataset.get("delta_stability") if isinstance(dataset.get("delta_stability"), dict) else {}
    manifest_delta_stability = manifest.get("delta_stability") if isinstance(manifest.get("delta_stability"), dict) else {}
    delta_status = str(delta_stability.get("status") or manifest_delta_stability.get("status") or "").strip().lower()
    delta_available = truthy(manifest.get("delta_sensitivity_study_available")) or truthy(manifest.get("delta_sensitivity_study_passed"))
    if not delta_available:
        delta_available = delta_status == "available"
    convergence_status = str(
        manifest.get("delta_stability_convergence_status")
        or delta_stability.get("delta_stability_convergence_status")
        or ""
    ).strip().lower()
    converged_value = manifest.get("delta_stability_converged")
    if converged_value is None:
        converged_value = delta_stability.get("delta_stability_converged")
    delta_converged = bool(converged_value) if converged_value is not None else False
    reference_noise = manifest.get("reference_noise") if isinstance(manifest.get("reference_noise"), dict) else {}
    reference_noise_status = str(reference_noise.get("status") or manifest.get("reference_noise_status") or "").strip().lower()
    comparison_statuses = unique_nonempty([row.get("comparison_status") for row in matrix_rows])
    return {
        "has_non_diagnostic_comparison_rows": bool(comparison_statuses) and all(status != "diagnostic_only" for status in comparison_statuses),
        "basis_gauge_verified": truthy(manifest.get("basis_gauge_verified")) or truthy(manifest.get("basis_gauge_evidence")),
        "orbital_ordering_verified": truthy(manifest.get("orbital_ordering_verified")) or truthy(manifest.get("orbital_ordering_evidence")),
        "delta_sensitivity_study_available": delta_available,
        "delta_sensitivity_study_passed": delta_available,
        "delta_stability_converged": delta_converged,
        "delta_stability_convergence_status": convergence_status or "not_evaluated_without_thresholds",
        "reference_noise_evidence": truthy(manifest.get("reference_noise_verified"))
        or truthy(manifest.get("reference_noise_evidence"))
        or reference_noise_status == "available",
        "independent_dataset_metadata": truthy(manifest.get("independent_dataset_metadata"))
        or truthy(manifest.get("dataset_split_evidence"))
        or (str(manifest.get("split") or "").strip().lower() == "test" and truthy(manifest.get("split_metadata_verified"))),
        "cross_model_equivalence_proven": truthy(manifest.get("cross_model_equivalence_proven"))
        or str(manifest.get("cross_model_equivalence_status") or "").strip().lower() == "proven",
        "paper_level_candidate_requested": truthy(manifest.get("paper_level_candidate_requested"))
        or truthy(manifest.get("paper_level_evidence_complete")),
    }


def evaluate_dataset(
    dataset: dict[str, Any],
    *,
    hermiticity_threshold: float,
    support_discontinuity_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    manifest = dataset["manifest"]
    model = dataset["model"]
    evidence_paths = list(dataset["paths"].values())
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    central_stencils = central_stencil_rows(dataset)
    central_metrics = central_metric_rows(dataset)

    if truthy(manifest.get("force_constants_used")):
        blockers.append(
            gate_row(
                "force_constants_used",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="SIESTA force constants must not be used as the dH/dR reference.",
                evidence_paths=evidence_paths,
            )
        )
    if str(manifest.get("reference_definition") or "").strip() != EXPECTED_REFERENCE_DEFINITION:
        blockers.append(
            gate_row(
                "reference_definition_invalid",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="reference_definition must be siesta_hamiltonian_finite_difference.",
                evidence_paths=evidence_paths,
            )
        )
    if not central_stencils and not central_metrics:
        blockers.append(
            gate_row(
                "missing_central_stencil",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="No central finite-difference stencil is available for derivative gating.",
                evidence_paths=evidence_paths,
            )
        )
    else:
        for row in central_stencils:
            if not str(row.get("plus_sample_id") or "").strip() or not str(row.get("minus_sample_id") or "").strip():
                blockers.append(
                    gate_row(
                        "missing_plus_minus_pairing",
                        model=model,
                        severity="blocker",
                        status="fail",
                        blocks_status="blocked",
                        message="Central stencils require both plus and minus samples.",
                        evidence_paths=evidence_paths,
                    )
                )
                break
    derivative_units = unique_nonempty([manifest.get("derivative_units"), *(row.get("derivative_units") for row in dataset["matrix_rows"])])
    if any(unit != EXPECTED_DERIVATIVE_UNITS for unit in derivative_units):
        blockers.append(
            gate_row(
                "mismatched_units",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message=f"Derivative units must be {EXPECTED_DERIVATIVE_UNITS}.",
                evidence_paths=evidence_paths,
            )
        )
    elif explicit_issue(dataset, ("unit_mismatch", "units mismatch", "hamiltonian_units", "displacement_units", "derivative_units")):
        blockers.append(
            gate_row(
                "mismatched_units",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="Unit mismatch evidence was found in derivative validation diagnostics.",
                evidence_paths=evidence_paths,
            )
        )
    delta_values = [number(row.get("delta_ang")) for row in central_metrics or central_stencils]
    if not delta_values or any(value is None or value <= 0 for value in delta_values):
        blockers.append(
            gate_row(
                "mismatched_delta",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="Central derivative rows must carry a positive delta_ang.",
                evidence_paths=evidence_paths,
            )
        )
    elif explicit_issue(dataset, ("delta_mismatch", "invalid_delta", "delta_ang")):
        blockers.append(
            gate_row(
                "mismatched_delta",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="Derivative validation reported mismatched or invalid delta metadata.",
                evidence_paths=evidence_paths,
            )
        )
    atom_indices = [int_or_none(row.get("atom_index_zero_based")) for row in central_metrics or central_stencils]
    if not atom_indices or any(index is None or index < 0 for index in atom_indices):
        blockers.append(
            gate_row(
                "atom_indexing_missing_or_inconsistent",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="atom_index_zero_based is missing or inconsistent in derivative rows.",
                evidence_paths=evidence_paths,
            )
        )
    elif explicit_issue(dataset, ("invalid_atom_index", "atom_index_mismatch", "atom_index_zero_based")):
        blockers.append(
            gate_row(
                "atom_indexing_missing_or_inconsistent",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="Derivative validation reported atom indexing inconsistencies.",
                evidence_paths=evidence_paths,
            )
        )
    if explicit_issue(dataset, ("shape_mismatch", "matrix shape", "shapes disagree", "matching shapes")):
        blockers.append(
            gate_row(
                "mismatched_shapes",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="Derivative validation reported mismatched matrix shapes.",
                evidence_paths=evidence_paths,
            )
        )
    if explicit_issue(dataset, ("orbital_ordering", "missing_required_metadata")):
        blockers.append(
            gate_row(
                "orbital_ordering_metadata_missing_or_inconsistent",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="Derivative validation reported missing or inconsistent orbital ordering metadata.",
                evidence_paths=evidence_paths,
            )
        )
    max_herm = max_hermiticity_defect(dataset)
    if max_herm > hermiticity_threshold:
        blockers.append(
            gate_row(
                "high_hermiticity_defect",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message=f"Hermiticity defect {max_herm:.3e} exceeds the threshold {hermiticity_threshold:.3e}.",
                evidence_paths=evidence_paths,
            )
        )
    if support_discontinuity_detected(dataset, support_discontinuity_threshold):
        blockers.append(
            gate_row(
                "support_pattern_discontinuity",
                model=model,
                severity="blocker",
                status="fail",
                blocks_status="blocked",
                message="Support pattern discontinuity or false-zero/false-nonzero activity exceeded the configured threshold.",
                evidence_paths=evidence_paths,
            )
        )
    if explicit_issue(dataset, ("neighbor_list_hash", "neighbor list", "sparsity_pattern_hash", "sparsity pattern")):
        warnings.append(
            warning_row(
                "neighbor_or_sparsity_warning",
                model=model,
                severity="severe",
                message="Neighbor-list or sparsity diagnostics were reported; interpret derivative comparisons conservatively.",
                evidence_paths=evidence_paths,
            )
        )
    if truthy(manifest.get("diagnostic_only_requested")) or str(manifest.get("scientific_status") or "").strip().lower() == "diagnostic_only":
        warnings.append(
            warning_row(
                "diagnostic_only_requested",
                model=model,
                message="This derivative evaluation was marked diagnostic-only in its manifest.",
                evidence_paths=evidence_paths,
            )
        )
    for fatal in manifest.get("fatal_errors") or []:
        if isinstance(fatal, dict):
            warnings.append(
                warning_row(
                    f"fatal_error_{fatal.get('kind') or 'reported'}",
                    model=model,
                    severity="severe",
                    message=str(fatal.get("message") or fatal.get("kind") or "Derivative fatal error reported."),
                    evidence_paths=evidence_paths,
                )
            )
    return blockers, warnings, dataset_paper_evidence(dataset)


def overall_status(
    datasets: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    paper_evidence: dict[str, bool],
) -> str:
    if any(row.get("blocks_status") == "blocked" for row in blockers):
        return "blocked"
    if any(
        truthy(dataset["manifest"].get("diagnostic_only_requested"))
        or str(dataset["manifest"].get("scientific_status") or "").strip().lower() == "diagnostic_only"
        for dataset in datasets
    ):
        return "internal_diagnostic"
    if any(not evidence["has_non_diagnostic_comparison_rows"] for evidence in paper_evidence.values()):
        return "internal_diagnostic"
    paper_blocked = [
        not info["basis_gauge_verified"]
        or not info["orbital_ordering_verified"]
        or not info["delta_stability_converged"]
        or not info["reference_noise_evidence"]
        or not info["independent_dataset_metadata"]
        for info in paper_evidence.values()
    ]
    if len(datasets) > 1 and not all(info["cross_model_equivalence_proven"] for info in paper_evidence.values()):
        paper_blocked.append(True)
    if not all(info["paper_level_candidate_requested"] for info in paper_evidence.values()):
        paper_blocked.append(True)
    if any(paper_blocked):
        return "technical_presentation"
    return "paper_level_candidate"


def allowed_claims_for_status(status: str) -> list[str]:
    if status == "blocked":
        return ["No scientific derivative comparison claim is allowed; only blocker diagnostics may be discussed."]
    claims = [
        "dH/dR refers to derivatives of Hamiltonian matrix elements with respect to Cartesian atomic displacement.",
        "The reference is finite differences of SIESTA Hamiltonians, not force constants.",
    ]
    if status == "internal_diagnostic":
        claims.append("Derivative errors may be shown as internal diagnostic-only evidence without ranking or winner claims.")
        return claims
    claims.append("Derivative metrics may be presented as technical finite-difference diagnostics against SIESTA Hamiltonians.")
    if status == "paper_level_candidate":
        claims.append("Paper-level candidate wording is allowed only as a candidate status pending independent review.")
    else:
        claims.append("No paper-level or winner claim is allowed from this report.")
    return claims


def blocked_claims_for_status(status: str, *, multiple_models: bool, paper_evidence: dict[str, bool]) -> list[str]:
    claims = [
        "Do not state or imply that SIESTA force constants, dynamical matrices, or phonons were used as the dH/dR reference.",
        "Do not declare a derivative winner by default.",
    ]
    if status in {"blocked", "internal_diagnostic", "technical_presentation"}:
        claims.append("Do not claim paper-level validation of Hamiltonian derivatives.")
    if status in {"blocked", "internal_diagnostic"}:
        claims.append("Do not frame the derivative comparison as more than diagnostic evidence.")
    if multiple_models and not all(info["cross_model_equivalence_proven"] for info in paper_evidence.values()):
        claims.append("Do not claim Graph2Mat vs DeepH paper-level comparability; cross-model equivalence remains diagnostic-only.")
    return claims


def recommended_next_steps(
    status: str,
    blockers: list[dict[str, Any]],
    paper_evidence: dict[str, bool],
    *,
    multiple_models: bool,
) -> list[str]:
    steps: list[str] = []
    blocker_ids = {row["id"] for row in blockers}
    if "force_constants_used" in blocker_ids:
        steps.append("Replace any force-constant reference with finite differences of SIESTA Hamiltonians.")
    if "missing_central_stencil" in blocker_ids or "missing_plus_minus_pairing" in blocker_ids:
        steps.append("Generate complete central plus/minus stencils before claiming derivative comparability.")
    if "mismatched_shapes" in blocker_ids or "mismatched_delta" in blocker_ids or "mismatched_units" in blocker_ids:
        steps.append("Repair derivative metadata consistency for shapes, delta, and units.")
    if "atom_indexing_missing_or_inconsistent" in blocker_ids:
        steps.append("Repair atom_index_zero_based metadata and ensure it is stable across all derivative rows.")
    if "orbital_ordering_metadata_missing_or_inconsistent" in blocker_ids:
        steps.append("Provide consistent orbital ordering metadata and fail closed until it is verified.")
    if "high_hermiticity_defect" in blocker_ids:
        steps.append("Lower derivative hermiticity defect below the configured threshold before technical presentation.")
    if "support_pattern_discontinuity" in blocker_ids:
        steps.append("Investigate false-zero/false-nonzero support changes and neighbor-list discontinuities.")
    if status != "blocked":
        for model, info in paper_evidence.items():
            if not info["basis_gauge_verified"]:
                steps.append(f"{model}: add explicit basis/gauge evidence before paper-level use.")
            if not info["orbital_ordering_verified"]:
                steps.append(f"{model}: add orbital-ordering evidence before paper-level use.")
            if not info["delta_sensitivity_study_available"]:
                steps.append(f"{model}: run or record a delta sensitivity study for paper-level gating.")
            elif not info["delta_stability_converged"]:
                steps.append(f"{model}: document delta stability convergence thresholds before paper-level use.")
            if not info["reference_noise_evidence"]:
                steps.append(f"{model}: add repeated-reference noise evidence before paper-level use.")
            if not info["independent_dataset_metadata"]:
                steps.append(f"{model}: add independent dataset/split metadata for paper-level gating.")
            if multiple_models and not info["cross_model_equivalence_proven"]:
                steps.append(f"{model}: keep cross-model claims diagnostic-only until equivalence evidence is proven.")
    return unique_nonempty(steps)


def discover_roots(run_root: Path | None, explicit_roots: list[Path]) -> list[Path]:
    roots = [normalize_root(root) for root in explicit_roots]
    if run_root is not None:
        for relative in (
            Path("common_metrics/graph2mat_eval/derivative_metrics"),
            Path("common_metrics/deeph_eval/derivative_metrics"),
        ):
            candidate = Path(run_root) / relative
            if (candidate / "manifest.json").exists():
                roots.append(candidate)
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    if not deduped:
        raise RuntimeError("No derivative_metrics roots were provided or discovered.")
    return deduped


def default_output_path(run_root: Path | None, roots: list[Path]) -> Path:
    if run_root is not None:
        return Path(run_root) / "common_metrics" / "summary" / "derivative_gate_report.json"
    first = roots[0]
    return first / "derivative_gate_report.json"


def build_derivative_gate_report(
    *,
    derivative_roots: list[Path],
    run_root: Path | None = None,
    hermiticity_threshold: float = DEFAULT_HERMITICITY_THRESHOLD,
    support_discontinuity_threshold: float = DEFAULT_SUPPORT_DISCONTINUITY_THRESHOLD,
) -> dict[str, Any]:
    datasets = [load_dataset(root) for root in derivative_roots]
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    paper_evidence: dict[str, bool] = {}
    for dataset in datasets:
        dataset_blockers, dataset_warnings, evidence = evaluate_dataset(
            dataset,
            hermiticity_threshold=hermiticity_threshold,
            support_discontinuity_threshold=support_discontinuity_threshold,
        )
        blockers.extend(dataset_blockers)
        warnings.extend(dataset_warnings)
        paper_evidence[dataset["model"]] = evidence

    status = overall_status(datasets, blockers, warnings, paper_evidence)
    if len(datasets) > 1 and not all(info["cross_model_equivalence_proven"] for info in paper_evidence.values()):
        blockers.append(
            gate_row(
                "cross_model_equivalence_diagnostic_only",
                severity="blocker",
                status="fail",
                blocks_status="paper_level_candidate",
                claim_scope="paper_level_only",
                message="Graph2Mat/DeepH cross-model equivalence remains diagnostic-only without explicit proof.",
                evidence_paths=[dataset["paths"]["manifest"] for dataset in datasets],
            )
        )
    for model, info in paper_evidence.items():
        if not info["delta_sensitivity_study_available"]:
            blockers.append(
                gate_row(
                    "paper_level_delta_sweep_missing",
                    model=model,
                    severity="blocker",
                    status="fail",
                    blocks_status="paper_level_candidate",
                    claim_scope="paper_level_only",
                    message="Paper-level candidate status requires a delta sensitivity study.",
                    evidence_paths=[next(dataset for dataset in datasets if dataset["model"] == model)["paths"]["manifest"]],
                )
            )
        elif not info["delta_stability_converged"]:
            blockers.append(
                gate_row(
                    "paper_level_delta_stability_not_converged",
                    model=model,
                    severity="blocker",
                    status="fail",
                    blocks_status="paper_level_candidate",
                    claim_scope="paper_level_only",
                    message="Paper-level candidate status requires documented delta stability convergence thresholds and convergence evidence.",
                    evidence_paths=[next(dataset for dataset in datasets if dataset["model"] == model)["paths"]["manifest"]],
                )
            )
        if not info["basis_gauge_verified"] or not info["orbital_ordering_verified"]:
            blockers.append(
                gate_row(
                    "paper_level_ordering_or_gauge_evidence_missing",
                    model=model,
                    severity="blocker",
                    status="fail",
                    blocks_status="paper_level_candidate",
                    claim_scope="paper_level_only",
                    message="Paper-level candidate status requires explicit orbital-ordering and basis/gauge evidence.",
                    evidence_paths=[next(dataset for dataset in datasets if dataset["model"] == model)["paths"]["manifest"]],
                )
            )
        if not info["reference_noise_evidence"]:
            blockers.append(
                gate_row(
                    "paper_level_reference_noise_missing",
                    model=model,
                    severity="blocker",
                    status="fail",
                    blocks_status="paper_level_candidate",
                    claim_scope="paper_level_only",
                    message="Paper-level candidate status requires repeated-reference/noise evidence.",
                    evidence_paths=[next(dataset for dataset in datasets if dataset["model"] == model)["paths"]["manifest"]],
                )
            )
        if not info["independent_dataset_metadata"]:
            blockers.append(
                gate_row(
                    "paper_level_independent_dataset_metadata_missing",
                    model=model,
                    severity="blocker",
                    status="fail",
                    blocks_status="paper_level_candidate",
                    claim_scope="paper_level_only",
                    message="Paper-level candidate status requires independent dataset/split metadata.",
                    evidence_paths=[next(dataset for dataset in datasets if dataset["model"] == model)["paths"]["manifest"]],
                )
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": status,
        "allowed_claims": allowed_claims_for_status(status),
        "blocked_claims": blocked_claims_for_status(status, multiple_models=len(datasets) > 1, paper_evidence=paper_evidence),
        "blockers": blockers,
        "warnings": warnings,
        "recommended_next_steps": recommended_next_steps(
            status,
            blockers,
            paper_evidence,
            multiple_models=len(datasets) > 1,
        ),
        "evidence_paths": {
            "run_root": str(run_root) if run_root is not None else "",
            "derivative_roots": [str(dataset["root"]) for dataset in datasets],
            "manifests": [str(dataset["paths"]["manifest"]) for dataset in datasets],
            "matrix_metrics": [str(dataset["paths"]["matrix_metrics"]) for dataset in datasets],
            "stencil_status": [str(dataset["paths"]["stencil_status"]) for dataset in datasets],
            "hermiticity": [str(dataset["paths"]["hermiticity"]) for dataset in datasets],
            "delta_stability": [str(dataset["paths"]["delta_stability"]) for dataset in datasets],
        },
        "datasets": [
            {
                "model": dataset["model"],
                "root": str(dataset["root"]),
                "manifest_scientific_status": dataset["manifest"].get("scientific_status"),
                "stencils_total": dataset["manifest"].get("stencils_total"),
                "stencils_ok": dataset["manifest"].get("stencils_ok"),
                "stencils_failed": dataset["manifest"].get("stencils_failed"),
                "paper_evidence": paper_evidence[dataset["model"]],
            }
            for dataset in datasets
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivative-root", dest="derivative_roots", action="append", type=Path, default=[])
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hermiticity-threshold", type=float, default=DEFAULT_HERMITICITY_THRESHOLD)
    parser.add_argument("--support-discontinuity-threshold", type=float, default=DEFAULT_SUPPORT_DISCONTINUITY_THRESHOLD)
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero when the generated report scientific_status is blocked.",
    )
    parser.add_argument(
        "--fail-below",
        choices=("internal_diagnostic", "technical_presentation", "paper_level_candidate"),
        default=None,
        help=(
            "Exit non-zero when the generated report scientific_status is below the requested minimum "
            "(blocked is always below these thresholds)."
        ),
    )
    return parser.parse_args()


def exit_code_for_report(
    report: dict[str, Any],
    *,
    output_path: Path,
    fail_on_blocked: bool = False,
    fail_below: str | None = None,
) -> tuple[int, str | None]:
    scientific_status = str(report.get("scientific_status") or "").strip()
    if scientific_status not in STATUS_RANK:
        return 1, (
            f"Derivative gate report at {output_path} has unknown scientific_status="
            f"{scientific_status or '<empty>'}."
        )
    if fail_on_blocked and scientific_status == "blocked":
        return 2, f"Derivative gate report at {output_path} has scientific_status=blocked."
    if fail_below is not None and STATUS_RANK[scientific_status] < STATUS_RANK[fail_below]:
        return 2, (
            f"Derivative gate report at {output_path} has scientific_status={scientific_status}, "
            f"below required minimum {fail_below}."
        )
    return 0, None


def main() -> int:
    args = parse_args()
    roots = discover_roots(args.run_root, args.derivative_roots)
    report = build_derivative_gate_report(
        derivative_roots=roots,
        run_root=args.run_root,
        hermiticity_threshold=float(args.hermiticity_threshold),
        support_discontinuity_threshold=float(args.support_discontinuity_threshold),
    )
    output_path = args.output or default_output_path(args.run_root, roots)
    write_json(output_path, report)
    print(json.dumps(json_safe(report), ensure_ascii=True))
    exit_code, error_message = exit_code_for_report(
        report,
        output_path=output_path,
        fail_on_blocked=bool(args.fail_on_blocked),
        fail_below=args.fail_below,
    )
    if error_message:
        print(error_message)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
