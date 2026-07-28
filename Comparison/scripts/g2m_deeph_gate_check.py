#!/usr/bin/env python3
"""Fail-closed robust-claim gate checker for Graph2Mat-vs-DeepH benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from deeph_prediction_adapter import (
    EQUIVALENCE_PROVEN_RAW_GLOBAL,
    EQUIVALENCE_SCOPE_RAW_GLOBAL,
    EQUIVALENCE_STATUS_PROVEN,
)
from g2m_deeph_protocol import protocol_hash, validate_protocol
from g2m_deeph_training_sweep import json_safe
from electronic_convergence import pooling_errors
from joint_artifact_contract import (
    material_profile_errors,
    md_temporal_evidence_errors,
    validate_recorded_snapshots,
)
from reference_selection import choose_reference_matrix
from run_inventory import collect_run_inventory


GATE_STATUS_SCHEMA = "graph2mat_deeph_gate_status_v1"
FORBIDDEN_REFERENCE_NAME = "ML_prediction.HSX"
CLAIM_STATUS_PRIORITY = (
    "invalid_protocol",
    "invalid_missing_evidence",
    "invalid_dataset",
    "invalid_equivalence",
    "invalid_final_statistics",
    "invalid_telemetry",
    "diagnostic_only",
)


def read_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"malformed JSON file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON payload must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def gate(
    gate_id: str,
    status: str,
    *,
    severity: str,
    message: str,
    evidence_paths: list[Path | str] | None = None,
    claim_status: str | None = None,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": status,
        "severity": severity,
        "claim_status": claim_status,
        "evidence_paths": [str(path) for path in (evidence_paths or []) if str(path)],
        "message": message,
    }


def pass_gate(gate_id: str, message: str, evidence_paths: list[Path | str] | None = None) -> dict[str, Any]:
    return gate(gate_id, "pass", severity="info", message=message, evidence_paths=evidence_paths)


def fail_gate(
    gate_id: str,
    message: str,
    *,
    claim_status: str,
    evidence_paths: list[Path | str] | None = None,
) -> dict[str, Any]:
    return gate(
        gate_id,
        "fail",
        severity="blocker",
        message=message,
        evidence_paths=evidence_paths,
        claim_status=claim_status,
    )


def candidate_final_statistics_paths(workflow_root: Path | None, run_root: Path | None) -> list[Path]:
    paths: list[Path] = []
    if workflow_root is not None:
        paths.append(workflow_root / "final_test" / "final_statistics.json")
    if run_root is not None:
        paths.append(run_root / "summary" / "final_statistics" / "final_statistics.json")
    return paths


def candidate_evidence_bundle_paths(workflow_root: Path | None) -> list[Path]:
    return [workflow_root / "evidence" / "evidence_bundle_manifest.json"] if workflow_root is not None else []


def find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def split_counts(split_manifest: dict[str, Any]) -> dict[str, int]:
    raw_counts = split_manifest.get("split_counts")
    if isinstance(raw_counts, dict):
        counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            try:
                counts[split] = int(raw_counts.get(split) or 0)
            except (TypeError, ValueError):
                counts[split] = 0
        return counts
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in split_manifest.get("rows") or []:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split") or "").strip()
        if split in counts:
            counts[split] += 1
    return counts


def forbidden_reference_findings(payload: Any, *, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}" if path else str(key)
            if "forbidden" in str(key).lower():
                continue
            findings.extend(forbidden_reference_findings(value, path=child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(forbidden_reference_findings(value, path=f"{path}[{index}]"))
    elif isinstance(payload, str) and FORBIDDEN_REFERENCE_NAME in payload:
        findings.append(f"{path}: {payload}")
    return findings


def validate_protocol_gate(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    raw_protocol = read_json_strict(protocol_path)
    try:
        protocol = validate_protocol(raw_protocol)
    except RuntimeError as exc:
        fallback = dict(raw_protocol)
        if "protocol_hash" not in fallback:
            try:
                fallback["protocol_hash"] = protocol_hash(fallback)
            except Exception:
                fallback["protocol_hash"] = ""
        return (
            fallback,
            raw_protocol,
            [
                fail_gate(
                    "protocol_valid",
                    str(exc),
                    claim_status="invalid_protocol",
                    evidence_paths=[protocol_path],
                )
            ],
        )
    return protocol, raw_protocol, [pass_gate("protocol_valid", "Protocol validates.", [protocol_path])]


def validate_dataset_gates(protocol: dict[str, Any], *, protocol_dir: Path) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    present_paths: list[Path] = []
    ready_failures: list[str] = []
    execution_failures: list[str] = []
    reference_failures: list[str] = []
    split_failures: list[str] = []
    forbidden_findings: list[str] = []
    pooling_datasets: list[tuple[Path, str | None]] = []
    for index, dataset in enumerate(protocol.get("datasets") or []):
        if not isinstance(dataset, dict):
            ready_failures.append(f"datasets[{index}] is not an object")
            continue
        dataset_id = str(dataset.get("dataset_id") or f"dataset_{index}")
        dataset_root = resolve_path(dataset.get("dataset_root"), base_dir=protocol_dir)
        benchmark_path = resolve_path(dataset.get("benchmark_dataset_manifest"), base_dir=protocol_dir)
        split_path = resolve_path(dataset.get("frozen_split_manifest"), base_dir=protocol_dir)
        artifact_path = resolve_path(
            dataset.get("artifact_validation") or dataset_root / "artifact_validation.json",
            base_dir=protocol_dir,
        )
        required_paths = [benchmark_path, split_path, artifact_path]
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            gates.append(
                fail_gate(
                    f"dataset_{dataset_id}_manifests_present",
                    "Dataset evidence is missing: " + ", ".join(str(path) for path in missing),
                    claim_status="invalid_missing_evidence",
                    evidence_paths=required_paths,
                )
            )
            continue
        present_paths.extend(required_paths)
        benchmark = read_json_strict(benchmark_path)
        pooling_datasets.append(
            (dataset_root, str(benchmark.get("material_label") or "") or None)
        )
        split = read_json_strict(split_path)
        artifact = read_json_strict(artifact_path)
        if benchmark.get("benchmark_ready") is not True or str(benchmark.get("validation_status") or "valid") == "invalid":
            ready_failures.append(f"{dataset_id}: benchmark_dataset_manifest is not benchmark_ready")
        if artifact.get("valid") is not True:
            ready_failures.append(f"{dataset_id}: artifact_validation.valid is not true")
        live_results, live_errors = validate_recorded_snapshots(
            artifact,
            base_dir=dataset_root,
        )
        if live_errors or not live_results:
            execution_failures.extend(
                f"{dataset_id}: {error}" for error in (live_errors or ["no snapshots revalidated"])
            )
        for result in live_results:
            selection = choose_reference_matrix(result.snapshot_dir)
            if not selection.ok:
                reference_failures.append(
                    f"{dataset_id}: {result.snapshot_dir}: {selection.reason}"
                )
        execution_failures.extend(
            f"{dataset_id}: {error}"
            for error in md_temporal_evidence_errors(dataset_root)
        )
        execution_failures.extend(
            f"{dataset_id}: {error}"
            for error in material_profile_errors(dataset_root, benchmark)
        )
        if split.get("valid") is False:
            split_failures.append(f"{dataset_id}: frozen_split_manifest.valid is false")
        counts = split_counts(split)
        missing_splits = [name for name, count in sorted(counts.items()) if count <= 0]
        if missing_splits:
            split_failures.append(f"{dataset_id}: empty split(s): {', '.join(missing_splits)}")
        for label, payload in (
            (f"{dataset_id}:benchmark_dataset_manifest", benchmark),
            (f"{dataset_id}:frozen_split_manifest", split),
            (f"{dataset_id}:artifact_validation", artifact),
        ):
            for finding in forbidden_reference_findings(payload):
                forbidden_findings.append(f"{label}:{finding}")
    if present_paths:
        gates.append(pass_gate("dataset_manifests_present", "Dataset manifest files are present.", present_paths))
    if ready_failures:
        gates.append(
            fail_gate(
                "dataset_benchmark_ready",
                "; ".join(ready_failures),
                claim_status="invalid_dataset",
                evidence_paths=present_paths,
            )
        )
    elif present_paths:
        gates.append(pass_gate("dataset_benchmark_ready", "Dataset manifests are benchmark_ready.", present_paths))
    if execution_failures:
        gates.append(
            fail_gate(
                "dataset_siesta_execution_valid",
                "; ".join(execution_failures),
                claim_status="invalid_dataset",
                evidence_paths=present_paths,
            )
        )
    if reference_failures:
        gates.append(
            fail_gate(
                "dataset_positive_reference_provenance",
                "; ".join(reference_failures),
                claim_status="invalid_dataset",
                evidence_paths=present_paths,
            )
        )
    elif present_paths:
        gates.append(
            pass_gate(
                "dataset_positive_reference_provenance",
                "Every selected SIESTA reference has matching positive hash provenance.",
                present_paths,
            )
        )
    elif present_paths:
        gates.append(
            pass_gate(
                "dataset_siesta_execution_valid",
                "Recorded snapshots pass live fail-closed SIESTA execution validation.",
                present_paths,
            )
        )
    if split_failures:
        gates.append(
            fail_gate(
                "frozen_split_nonempty",
                "; ".join(split_failures),
                claim_status="invalid_dataset",
                evidence_paths=present_paths,
            )
        )
    elif present_paths:
        gates.append(pass_gate("frozen_split_nonempty", "Frozen splits have non-empty train/validation/test rows.", present_paths))
    if forbidden_findings:
        gates.append(
            fail_gate(
                "forbidden_reference_absent",
                "Forbidden reference artifact detected: " + "; ".join(forbidden_findings[:10]),
                claim_status="invalid_dataset",
                evidence_paths=present_paths,
            )
        )
    elif present_paths:
        gates.append(pass_gate("forbidden_reference_absent", "No ML_prediction.HSX reference paths found.", present_paths))
    pooling_failures = pooling_errors(pooling_datasets)
    if pooling_failures:
        gates.append(
            fail_gate(
                "electronic_convergence_for_pooling",
                "; ".join(pooling_failures),
                claim_status="invalid_dataset",
                evidence_paths=present_paths,
            )
        )
    elif len(pooling_datasets) > 1:
        gates.append(
            pass_gate(
                "electronic_convergence_for_pooling",
                "Cross-dataset pooling has matching material-specific convergence evidence.",
                present_paths,
            )
        )
    return gates


def validate_selection_gate(protocol: dict[str, Any]) -> dict[str, Any]:
    selection = protocol.get("selection") if isinstance(protocol.get("selection"), dict) else {}
    top_k = protocol.get("top_k_selection") if isinstance(protocol.get("top_k_selection"), dict) else {}
    if (
        selection.get("split") == "validation"
        and selection.get("source") == "validation_only"
        and top_k.get("split") == "validation"
        and top_k.get("uses_test_metrics") is False
    ):
        return pass_gate("selection_validation_only", "Selection and top-k are validation-only.")
    return fail_gate(
        "selection_validation_only",
        "Selection/top-k policy is not validation-only.",
        claim_status="invalid_protocol",
    )


def validate_blindness_artifact_gates(workflow_root: Path | None) -> list[dict[str, Any]]:
    if workflow_root is None:
        return [
            fail_gate(
                "test_blindness_artifacts",
                "workflow_root is required to verify validation → freeze → final_test chronology.",
                claim_status="invalid_missing_evidence",
            )
        ]
    selected_path = workflow_root / "selection" / "selected_configs.json"
    plan_path = workflow_root / "selection" / "robust_rerun_plan.json"
    final_path = workflow_root / "final_test" / "run_final_test_manifest.json"
    paths = [selected_path, plan_path, final_path]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return [
            fail_gate(
                "test_blindness_artifacts",
                "Missing test-blind chronology artifacts: " + ", ".join(missing),
                claim_status="invalid_missing_evidence",
                evidence_paths=paths,
            )
        ]
    selected = read_json_strict(selected_path)
    plan = read_json_strict(plan_path)
    final = read_json_strict(final_path)
    blockers: list[str] = []
    if selected.get("protocol_stage") != "search" or selected.get("uses_test_metrics") is not False:
        blockers.append("selection_not_validation_only")
    selected_configs = [
        row for row in selected.get("selected_configs") or [] if isinstance(row, dict)
    ]
    if not selected_configs:
        blockers.append("selected_frozen_candidates_missing")
    if selected.get("checkpoint_selection_complete") is not True or any(
        (row.get("checkpoint_selection") or {}).get("status") != "valid"
        for row in selected_configs
    ):
        blockers.extend(str(item) for item in selected.get("paper_level_blockers") or ["checkpoint_selection_incomplete"])
    if plan.get("protocol_stage") != "robust_validation" or plan.get("uses_test_metrics") is not False:
        blockers.append("candidate_freeze_not_test_blind")
    blockers.extend(str(item) for item in plan.get("paper_level_blockers") or [])
    if final.get("status") != "completed":
        blockers.append("final_test_not_completed_after_candidate_freeze")
    planned = {
        (str(row.get("model") or ""), str(row.get("config_id") or ""))
        for row in plan.get("planned_runs") or []
        if isinstance(row, dict)
    }
    final_rows = final.get("evaluated_runs") or []
    evaluated = {
        (str(row.get("model") or ""), str(row.get("config_id") or ""))
        for row in final_rows
        if isinstance(row, dict)
    }
    if not planned or not evaluated or not evaluated.issubset(planned):
        blockers.append("final_test_contains_non_frozen_or_unverifiable_candidates")
    if blockers:
        return [
            fail_gate(
                "test_blindness_artifacts",
                "; ".join(sorted(set(blockers))),
                claim_status="invalid_final_statistics",
                evidence_paths=paths,
            )
        ]
    return [
        pass_gate(
            "test_blindness_artifacts",
            "Validation-only selection, frozen candidates and final-test chronology are verified.",
            paths,
        )
    ]


def load_final_statistics(path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        return {}, [
            fail_gate(
                "final_statistics_present",
                "final_statistics.json is missing.",
                claim_status="invalid_missing_evidence",
            )
        ]
    return read_json_strict(path), [pass_gate("final_statistics_present", "Final statistics manifest exists.", [path])]


def validate_final_statistics_gates(
    final_stats: dict[str, Any],
    *,
    final_stats_path: Path | None,
    expected_seed_count: int,
) -> list[dict[str, Any]]:
    if not final_stats:
        return []
    gates: list[dict[str, Any]] = []
    summary_rows = [row for row in final_stats.get("final_seed_summary") or [] if isinstance(row, dict)]
    seed_failures = []
    for row in summary_rows:
        completed = int(row.get("n_seeds_completed") or 0)
        model = str(row.get("model") or "unknown")
        dataset = str(row.get("dataset_id") or "dataset")
        if completed < expected_seed_count:
            seed_failures.append(f"{model}/{dataset}: {completed} < {expected_seed_count}")
    if not summary_rows:
        seed_failures.append("final_seed_summary is empty")
    if seed_failures:
        gates.append(
            fail_gate(
                "final_seeds_complete",
                "Incomplete final seeds: " + "; ".join(seed_failures),
                claim_status="invalid_final_statistics",
                evidence_paths=[final_stats_path] if final_stats_path else None,
            )
        )
    else:
        gates.append(pass_gate("final_seeds_complete", "Final seed counts meet the protocol requirement.", [final_stats_path] if final_stats_path else None))

    winners = final_stats.get("winner_decision") if isinstance(final_stats.get("winner_decision"), dict) else {}
    if winners.get("robust_claim_allowed") is True:
        gates.append(pass_gate("final_statistics_robust", "Final statistics allow a robust claim.", [final_stats_path] if final_stats_path else None))
    else:
        reason = "; ".join(str(item) for item in winners.get("gates_failed") or []) or str(
            winners.get("diagnostic_only_reason") or "winner_decision.robust_claim_allowed is not true"
        )
        gates.append(
            fail_gate(
                "final_statistics_robust",
                reason,
                claim_status="diagnostic_only" if "diagnostic" in reason else "invalid_final_statistics",
                evidence_paths=[final_stats_path] if final_stats_path else None,
            )
        )
    return gates


def validate_telemetry_gate(final_stats: dict[str, Any], *, final_stats_path: Path | None) -> dict[str, Any]:
    missing: list[str] = []
    for row in final_stats.get("final_seed_summary") or []:
        if not isinstance(row, dict):
            continue
        label = f"{row.get('model') or 'unknown'}/{row.get('dataset_id') or 'dataset'}"
        if row.get("gpu_hours_mean") is None:
            missing.append(f"{label}: gpu_hours_mean")
        if row.get("peak_gpu_memory_mb_mean") is None:
            missing.append(f"{label}: peak_gpu_memory_mb_mean")
    if not final_stats.get("final_seed_summary"):
        missing.append("final_seed_summary")
    if missing:
        return fail_gate(
            "telemetry_complete",
            "Missing telemetry fields: " + ", ".join(missing),
            claim_status="invalid_telemetry",
            evidence_paths=[final_stats_path] if final_stats_path else None,
        )
    return pass_gate("telemetry_complete", "GPU-hours and peak GPU memory are present.", [final_stats_path] if final_stats_path else None)


def evidence_bundle_path_from_args(workflow_root: Path | None) -> Path | None:
    return find_existing(candidate_evidence_bundle_paths(workflow_root))


def validate_evidence_bundle_gate(evidence_path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if evidence_path is None:
        return {}, [
            fail_gate(
                "evidence_bundle_complete",
                "evidence_bundle_manifest.json is missing.",
                claim_status="invalid_missing_evidence",
            )
        ]
    payload = read_json_strict(evidence_path)
    missing_required = list(payload.get("missing_required") or [])
    files = [entry for entry in payload.get("files") or [] if isinstance(entry, dict)]
    for entry in files:
        if entry.get("required") is True:
            path_text = str(entry.get("path") or "")
            if entry.get("exists") is not True:
                missing_required.append(str(entry.get("label") or path_text or "required_file"))
            elif path_text and not Path(path_text).exists():
                missing_required.append(str(entry.get("label") or path_text))
    if str(payload.get("status") or "") != "complete":
        missing_required.append(f"bundle status is {payload.get('status') or 'missing'}")
    if missing_required:
        return payload, [
            fail_gate(
                "evidence_bundle_complete",
                "Evidence bundle is incomplete: " + ", ".join(sorted(set(missing_required))),
                claim_status="invalid_missing_evidence",
                evidence_paths=[evidence_path],
            )
        ]
    return payload, [pass_gate("evidence_bundle_complete", "Evidence bundle is complete.", [evidence_path])]


def adapter_manifest_paths(
    *,
    run_root: Path | None,
    evidence_bundle: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    for entry in evidence_bundle.get("files") or []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "")
        path = str(entry.get("path") or "")
        if "adapter_manifest" in label and path:
            paths.append(Path(path))
    if run_root is not None and run_root.exists():
        paths.extend(sorted(run_root.rglob("adapter_manifest.json")))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def adapter_manifest_is_proven(payload: dict[str, Any]) -> bool:
    statuses = [str(item) for item in payload.get("adapter_equivalence_statuses") or [] if str(item)]
    equivalence_statuses = [str(item) for item in payload.get("equivalence_statuses") or [] if str(item)]
    scopes = [str(item) for item in payload.get("equivalence_scopes") or [] if str(item)]
    gate_payload = payload.get("equivalence_gate") if isinstance(payload.get("equivalence_gate"), dict) else {}
    return (
        bool(payload.get("robust_matrix_metrics_allowed"))
        and gate_payload.get("robust_claim_allowed") is True
        and bool(statuses)
        and all(status == EQUIVALENCE_PROVEN_RAW_GLOBAL for status in statuses)
        and bool(equivalence_statuses)
        and all(status == EQUIVALENCE_STATUS_PROVEN for status in equivalence_statuses)
        and bool(scopes)
        and all(scope == EQUIVALENCE_SCOPE_RAW_GLOBAL for scope in scopes)
    )


def validate_deeph_equivalence_gate(
    protocol: dict[str, Any],
    *,
    run_root: Path | None,
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    deeph = protocol.get("models", {}).get("deeph") if isinstance(protocol.get("models"), dict) else {}
    if not isinstance(deeph, dict) or deeph.get("enabled") is not True:
        return pass_gate("deeph_equivalence_proven", "DeepH is not enabled in this protocol.")
    paths = adapter_manifest_paths(run_root=run_root, evidence_bundle=evidence_bundle)
    if not paths:
        return fail_gate(
            "deeph_equivalence_proven",
            "No DeepH adapter_manifest.json was found.",
            claim_status="invalid_equivalence",
        )
    failures: list[str] = []
    for path in paths:
        if not path.exists():
            failures.append(f"{path}: missing")
            continue
        payload = read_json_strict(path)
        if not adapter_manifest_is_proven(payload):
            failures.append(f"{path}: raw/global equivalence is not proven")
    if failures:
        return fail_gate(
            "deeph_equivalence_proven",
            "; ".join(failures),
            claim_status="invalid_equivalence",
            evidence_paths=paths,
        )
    return pass_gate("deeph_equivalence_proven", "DeepH raw/global equivalence is proven.", paths)


def claim_status_from_gates(gates: list[dict[str, Any]]) -> str:
    failed_statuses = {
        str(gate.get("claim_status"))
        for gate in gates
        if gate.get("status") == "fail" and gate.get("claim_status")
    }
    if not failed_statuses:
        return "robust_allowed"
    for status in CLAIM_STATUS_PRIORITY:
        if status in failed_statuses:
            return status
    return "invalid_missing_evidence"


def required_next_actions(gates: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for item in gates:
        if item.get("status") != "fail":
            continue
        actions.append(f"{item.get('id')}: {item.get('message')}")
    return actions


def build_gate_status(
    *,
    protocol_path: Path,
    workflow_root: Path | None = None,
    run_root: Path | None = None,
    run_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workflow_root = Path(workflow_root) if workflow_root is not None else None
    run_root = Path(run_root) if run_root is not None else None
    protocol, _raw_protocol, gates = validate_protocol_gate(protocol_path)
    protocol_dir = protocol_path.parent
    if protocol.get("protocol_id"):
        gates.append(validate_selection_gate(protocol))
    gates.extend(validate_blindness_artifact_gates(workflow_root))
    gates.extend(validate_dataset_gates(protocol, protocol_dir=protocol_dir))

    final_stats_path = find_existing(candidate_final_statistics_paths(workflow_root, run_root))
    final_stats, final_stats_gates = load_final_statistics(final_stats_path)
    gates.extend(final_stats_gates)
    expected_seed_count = max(5, len(protocol.get("final_seeds") or []))
    gates.extend(
        validate_final_statistics_gates(
            final_stats,
            final_stats_path=final_stats_path,
            expected_seed_count=expected_seed_count,
        )
    )
    if final_stats:
        gates.append(validate_telemetry_gate(final_stats, final_stats_path=final_stats_path))

    evidence_bundle, evidence_gates = validate_evidence_bundle_gate(evidence_bundle_path_from_args(workflow_root))
    gates.extend(evidence_gates)
    gates.append(validate_deeph_equivalence_gate(protocol, run_root=run_root, evidence_bundle=evidence_bundle))
    run_inventory = run_inventory or collect_run_inventory(
        deeph_python=REPO_ROOT.parent / "DeepH-pack" / ".venv" / "bin" / "python",
    )
    reproducibility = str(run_inventory.get("reproducibility_status") or "unavailable")
    gates.append(
        pass_gate("reproducibility_pinned_clean", "All source repositories are pinned and clean.")
        if reproducibility == "pinned_clean"
        else fail_gate(
            "reproducibility_pinned_clean",
            f"reproducibility_status={reproducibility}; robust claims require pinned_clean repositories",
            claim_status="diagnostic_only",
        )
    )

    status = claim_status_from_gates(gates)
    blockers = [str(item.get("message") or "") for item in gates if item.get("status") == "fail"]
    warnings = [str(item.get("message") or "") for item in gates if item.get("status") == "warn"]
    return {
        "schema": GATE_STATUS_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "protocol_path": str(protocol_path),
        "workflow_root": str(workflow_root) if workflow_root is not None else "",
        "run_root": str(run_root) if run_root is not None else "",
        "run_inventory": run_inventory,
        "robust_claim_allowed": status == "robust_allowed",
        "diagnostic_only": status != "robust_allowed",
        "claim_status": status,
        "gates": gates,
        "blockers": blockers,
        "warnings": warnings,
        "required_next_actions": required_next_actions(gates),
    }


def error_status(*, protocol_path: Path | None, workflow_root: Path | None, run_root: Path | None, error: str) -> dict[str, Any]:
    gate_payload = fail_gate(
        "gate_check_error",
        error,
        claim_status="invalid_protocol" if protocol_path else "invalid_missing_evidence",
        evidence_paths=[protocol_path] if protocol_path else None,
    )
    return {
        "schema": GATE_STATUS_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_id": "",
        "protocol_hash": "",
        "protocol_path": str(protocol_path) if protocol_path is not None else "",
        "workflow_root": str(workflow_root) if workflow_root is not None else "",
        "run_root": str(run_root) if run_root is not None else "",
        "robust_claim_allowed": False,
        "diagnostic_only": True,
        "claim_status": gate_payload["claim_status"],
        "gates": [gate_payload],
        "blockers": [error],
        "warnings": [],
        "required_next_actions": [f"gate_check_error: {error}"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workflow-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_gate_status(
            protocol_path=args.protocol,
            workflow_root=args.workflow_root,
            run_root=args.run_root,
        )
    except RuntimeError as exc:
        payload = error_status(
            protocol_path=args.protocol,
            workflow_root=args.workflow_root,
            run_root=args.run_root,
            error=str(exc),
        )
        write_json(args.output, payload)
        print(str(exc), file=sys.stderr)
        return 1
    write_json(args.output, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
