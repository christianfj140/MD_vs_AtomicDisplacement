#!/usr/bin/env python3
"""Staged final/publicable workflow for the Graph2Mat-vs-DeepH benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from g2m_deeph_final_stats import final_statistics_report, load_rows, metric_split, metric_value, protocol_stage
from g2m_deeph_protocol import load_protocol
from g2m_deeph_report import generate_report
from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner
from g2m_deeph_test_blindness import (
    FINAL_TEST_STAGE,
    ROBUST_VALIDATION_STAGE,
    SEARCH_STAGE,
    assert_no_test_metrics_for_search,
    validate_final_evaluation_inputs,
)
from g2m_deeph_topk import (
    generate_robust_rerun_plan,
    select_top_configs,
    write_selection_artifacts,
)
from g2m_deeph_training_sweep import expand_training_sweep, json_safe, training_sweep_from_protocol
from g2m_deeph_verify_protocol_datasets import verify_protocol_datasets


WORKFLOW_SCHEMA = "graph2mat_deeph_final_workflow_v1"
EVIDENCE_BUNDLE_SCHEMA = "graph2mat_deeph_final_evidence_bundle_v1"
STAGES = {
    "validate-protocol",
    "generate-search-plan",
    "run-search",
    "select-top-k",
    "generate-final-seeds",
    "run-final",
    "run-final-test",
    "evaluate-final-test",
    "generate-report",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for row in rows:
        safe: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                safe[key] = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False)
            else:
                safe[key] = value
            if key not in fieldnames:
                fieldnames.append(key)
        safe_rows.append(safe)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(safe_rows)


def stage_path(workflow_root: Path, stage: str) -> Path:
    return workflow_root / "stages" / f"{stage}.json"


def stage_manifest(
    *,
    workflow_root: Path,
    stage: str,
    status: str,
    outputs: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    message: str = "",
) -> dict[str, Any]:
    manifest = {
        "schema": WORKFLOW_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workflow_root": str(workflow_root),
        "stage": stage,
        "status": status,
        "inputs": inputs or {},
        "outputs": outputs or {},
        "message": message,
    }
    write_json(stage_path(workflow_root, stage), manifest)
    write_json(workflow_root / "workflow_state.json", manifest)
    return manifest


def require_path(path: Path, *, message: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{message}: {path}")
    return path


def canonical_protocol_path(workflow_root: Path) -> Path:
    return workflow_root / "protocol" / "validated_protocol.json"


def load_workflow_protocol(protocol_path: Path | None, workflow_root: Path) -> dict[str, Any]:
    if protocol_path is not None:
        return load_protocol(protocol_path)
    return read_json(require_path(canonical_protocol_path(workflow_root), message="Missing validated protocol"))


def generate_training_plan(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    training_sweep = training_sweep_from_protocol(protocol)
    plan = expand_training_sweep(training_sweep, datasets=list(protocol.get("datasets") or []))
    return training_sweep, plan


def protocol_dataset_ids(protocol: dict[str, Any]) -> list[str]:
    return [str(dataset.get("dataset_id") or "").strip() for dataset in protocol.get("datasets") or []]


def selected_protocol_dataset(protocol: dict[str, Any], dataset_id: str | None) -> dict[str, Any]:
    datasets = [dict(dataset) for dataset in protocol.get("datasets") or [] if isinstance(dataset, dict)]
    if not datasets:
        raise RuntimeError("Protocol contains no datasets.")
    ids = [str(dataset.get("dataset_id") or "").strip() for dataset in datasets]
    requested = str(dataset_id or "").strip()
    if not requested:
        if len(datasets) == 1:
            return datasets[0]
        raise RuntimeError(
            "Protocol declares multiple datasets ("
            + ", ".join(ids)
            + "); pass --dataset-id for stages that launch a runner."
        )
    for dataset in datasets:
        if str(dataset.get("dataset_id") or "").strip() == requested:
            return dataset
    raise RuntimeError(f"Unknown protocol dataset_id {requested!r}. Available datasets: {', '.join(ids)}")


def filter_plan_for_dataset(plan: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    filtered = json.loads(json.dumps(json_safe(plan)))
    planned = [
        dict(record)
        for record in filtered.get("planned_runs") or []
        if str(record.get("dataset_id") or "") == dataset_id
    ]
    if not planned:
        raise RuntimeError(f"Training plan contains no runs for dataset_id={dataset_id!r}.")
    filtered["planned_runs"] = planned
    search_plan = filtered.get("search_plan")
    if isinstance(search_plan, dict):
        search_plan["planned_runs"] = [
            dict(record)
            for record in search_plan.get("planned_runs") or []
            if str(record.get("dataset_id") or "") == dataset_id
        ]
        search_plan["planned_run_count"] = len(search_plan["planned_runs"])
        duplicates = [
            dict(record)
            for record in search_plan.get("duplicate_configs") or []
            if str(record.get("dataset_id") or "") == dataset_id
        ]
        search_plan["duplicate_configs"] = duplicates
        search_plan["duplicate_config_count"] = len(duplicates)
    return filtered


def build_runner_payload(
    *,
    protocol: dict[str, Any],
    plan: dict[str, Any],
    workflow_root: Path,
    run_id: str,
    protocol_stage: str,
    dry_run: bool,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    selected_dataset = selected_protocol_dataset(protocol, dataset_id)
    selected_dataset_id = str(selected_dataset.get("dataset_id") or "").strip()
    selected_plan = filter_plan_for_dataset(plan, selected_dataset_id)
    payload: dict[str, Any] = {
        "benchmark_mode": "final_publication",
        "protocol_stage": protocol_stage,
        "protocol": protocol,
        "run_id": run_id,
        "output_root": str(workflow_root / "runs"),
        "dataset_root": str(selected_dataset["dataset_root"]),
        "selected_dataset_id": selected_dataset_id,
        "protocol_dataset_ids": protocol_dataset_ids(protocol),
        "executed_dataset_ids": [selected_dataset_id],
        "dataset_mode": "reuse_validated",
        "metric_fail_policy": "fail_closed",
        "allow_diagnostic_metrics": False,
        "dry_run": dry_run,
        "early_stopping": dict(protocol.get("early_stopping") or {}),
        "selection_metric": (protocol.get("selection") or {}).get("metric"),
        "selection_mode": (protocol.get("selection") or {}).get("mode"),
        "_training_sweep_plan": selected_plan,
    }
    if isinstance(protocol.get("performance"), dict):
        payload["performance"] = dict(protocol["performance"])
    if isinstance(protocol.get("deeph"), dict):
        payload["deeph"] = dict(protocol["deeph"])
    for key in ("pipeline_config", "md_pipeline_config"):
        if protocol.get(key) not in (None, ""):
            payload[key] = protocol[key]
    return payload


def wait_for_runner(
    *,
    runner: Graph2MatDeepHBenchmarkRunner,
    output_manifest: Path,
    poll_seconds: float,
) -> dict[str, Any]:
    while True:
        status = runner.status()
        if not status.get("running"):
            break
        run_root = Path(str(status.get("run_root") or ""))
        sweep_manifest_path = run_root / "sweep" / "training_sweep_manifest.json"
        if sweep_manifest_path.exists():
            sweep_manifest = read_json(sweep_manifest_path)
            sweep_status = str(sweep_manifest.get("status") or "")
            runs = sweep_manifest.get("runs") if isinstance(sweep_manifest.get("runs"), list) else []
            planned = sweep_manifest.get("planned_runs") if isinstance(sweep_manifest.get("planned_runs"), list) else []
            if sweep_status in {"completed", "completed_with_failures", "stopped"} and (
                not planned or len(runs) >= len(planned)
            ):
                break
        time.sleep(max(0.2, poll_seconds))
    runner.write_incremental_manifest(output_manifest)
    payload = read_json(output_manifest)
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    if int(status.get("returncode") or 0) != 0:
        raise RuntimeError(f"Graph2Mat/DeepH runner failed with return code {status.get('returncode')}: {status.get('error')}")
    return payload


def copy_if_exists(source: Path, destination: Path) -> str:
    if not source.exists():
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_file_entry(path: Path, *, label: str, required: bool = True) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "required": required,
        "exists": path.exists(),
        "sha256": sha256_file(path),
    }


def build_evidence_bundle_manifest(
    *,
    workflow_root: Path,
    protocol: dict[str, Any],
    run_root: Path,
    report_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for dataset in protocol.get("datasets") or []:
        dataset_id = str(dataset.get("dataset_id") or "dataset")
        dataset_root = Path(str(dataset.get("dataset_root") or ""))
        entries.extend(
            [
                evidence_file_entry(
                    Path(str(dataset.get("benchmark_dataset_manifest") or "")),
                    label=f"{dataset_id}:benchmark_dataset_manifest",
                ),
                evidence_file_entry(
                    Path(str(dataset.get("frozen_split_manifest") or "")),
                    label=f"{dataset_id}:frozen_split_manifest",
                ),
                evidence_file_entry(
                    dataset_root / "artifact_validation.json",
                    label=f"{dataset_id}:artifact_validation",
                ),
            ]
        )
    fixed_paths = {
        "validated_protocol": workflow_root / "protocol" / "validated_protocol.json",
        "search_plan": workflow_root / "search" / "search_plan.json",
        "search_training_sweep_manifest": workflow_root / "search" / "training_sweep_manifest.json",
        "selected_configs": workflow_root / "selection" / "selected_configs.json",
        "robust_rerun_plan": workflow_root / "selection" / "robust_rerun_plan.json",
        "final_training_sweep_manifest": workflow_root / "final" / "training_sweep_manifest.json",
        "final_statistics": workflow_root / "final_test" / "final_statistics.json",
        "run_training_sweep_manifest": run_root / "sweep" / "training_sweep_manifest.json",
        "ranking_summary": run_root / "summary" / "ranking" / "ranking_summary.json",
    }
    entries.extend(evidence_file_entry(path, label=label, required=False) for label, path in fixed_paths.items())
    for index, path in enumerate(sorted(run_root.rglob("telemetry/*.json"))):
        entries.append(evidence_file_entry(path, label=f"telemetry:{index}", required=False))
    for index, path in enumerate(sorted(run_root.rglob("adapter_manifest.json"))):
        entries.append(evidence_file_entry(path, label=f"deeph_adapter_manifest:{index}", required=False))
    for key, value in sorted((report_outputs or {}).items()):
        if isinstance(value, str) and value:
            entries.append(evidence_file_entry(Path(value), label=f"report:{key}", required=False))
    missing_required = [entry["label"] for entry in entries if entry.get("required") and not entry.get("exists")]
    payload = {
        "schema": EVIDENCE_BUNDLE_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workflow_root": str(workflow_root),
        "run_root": str(run_root),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "protocol_dataset_ids": protocol_dataset_ids(protocol),
        "status": "complete" if not missing_required else "partial",
        "missing_required": missing_required,
        "files": entries,
    }
    output = workflow_root / "evidence" / "evidence_bundle_manifest.json"
    write_json(output, payload)
    payload["path"] = str(output)
    return payload


def stage_validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_protocol(args.protocol)
    path = canonical_protocol_path(workflow_root)
    write_json(path, protocol)
    outputs: dict[str, Any] = {"validated_protocol": str(path), "protocol_hash": protocol["protocol_hash"]}
    if args.verify_datasets:
        verification_path = workflow_root / "dataset_verification.json"
        verification = verify_protocol_datasets(
            protocol_path=args.protocol,
            output_path=verification_path,
            strict=True,
            write_manifests=bool(args.write_dataset_manifests),
        )
        outputs["dataset_verification"] = str(verification_path)
        outputs["dataset_verification_status"] = verification.get("status")
        if verification.get("status") != "valid":
            raise RuntimeError("Protocol dataset verification failed: " + "; ".join(verification.get("blockers") or []))
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        inputs={"protocol": str(args.protocol)},
        outputs=outputs,
        message="Protocol validated and canonicalized.",
    )


def stage_generate_search_plan(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    training_sweep, plan = generate_training_plan(protocol)
    search_dir = workflow_root / "search"
    write_json(search_dir / "training_sweep_payload.json", training_sweep)
    write_json(search_dir / "training_sweep_plan.json", plan)
    write_json(search_dir / "search_plan.json", plan["search_plan"])
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        inputs={"protocol_hash": protocol.get("protocol_hash")},
        outputs={
            "training_sweep_payload": str(search_dir / "training_sweep_payload.json"),
            "training_sweep_plan": str(search_dir / "training_sweep_plan.json"),
            "search_plan": str(search_dir / "search_plan.json"),
            "planned_run_count": len(plan.get("planned_runs") or []),
            "protocol_dataset_ids": protocol_dataset_ids(protocol),
            "planned_dataset_ids": sorted({str(row.get("dataset_id") or "") for row in plan.get("planned_runs") or []}),
        },
        message="Preregistered search plan generated.",
    )


def stage_run_search(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    plan = read_json(require_path(workflow_root / "search" / "training_sweep_plan.json", message="Missing search plan"))
    run_id = args.run_id or f"{protocol['protocol_id']}_search"
    payload = build_runner_payload(
        protocol=protocol,
        plan=plan,
        workflow_root=workflow_root,
        run_id=run_id,
        protocol_stage=SEARCH_STAGE,
        dry_run=bool(args.dry_run),
        dataset_id=args.dataset_id,
    )
    expected_run_root = workflow_root / "runs" / run_id
    if expected_run_root.exists():
        payload["reuse_run_root"] = True
        payload["resume_training_sweep"] = True
        payload["resume_from_run_root"] = str(expected_run_root)
    selected_plan = payload["_training_sweep_plan"]
    search_dir = workflow_root / "search"
    write_json(search_dir / "run_search_payload.json", payload)
    if args.dry_run:
        manifest = {
            "schema": WORKFLOW_SCHEMA,
            "stage": "run-search",
            "status": "planned_dry_run",
            "run_id": run_id,
            "selected_dataset_id": payload["selected_dataset_id"],
            "executed_dataset_ids": payload["executed_dataset_ids"],
            "protocol_dataset_ids": payload["protocol_dataset_ids"],
            "runner_payload": str(search_dir / "run_search_payload.json"),
            "planned_run_count": len(selected_plan.get("planned_runs") or []),
        }
        write_json(search_dir / "run_search_manifest.json", manifest)
        return stage_manifest(
            workflow_root=workflow_root,
            stage=args.stage,
            status="planned_dry_run",
            outputs={
                "run_search_manifest": str(search_dir / "run_search_manifest.json"),
                "selected_dataset_id": payload["selected_dataset_id"],
                "executed_dataset_ids": payload["executed_dataset_ids"],
                "protocol_dataset_ids": payload["protocol_dataset_ids"],
                "planned_run_count": len(selected_plan.get("planned_runs") or []),
            },
            message="Search runner payload written; no subprocesses launched.",
        )
    runner = Graph2MatDeepHBenchmarkRunner()
    runner.start(payload)
    manifest = wait_for_runner(
        runner=runner,
        output_manifest=search_dir / "run_search_manifest.json",
        poll_seconds=float(args.poll_seconds),
    )
    status = manifest.get("status") if isinstance(manifest.get("status"), dict) else {}
    run_root = Path(str(status.get("run_root") or ""))
    copied = copy_if_exists(run_root / "sweep" / "training_sweep_manifest.json", search_dir / "training_sweep_manifest.json")
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        outputs={
            "run_search_manifest": str(search_dir / "run_search_manifest.json"),
            "search_run_root": str(run_root),
            "training_sweep_manifest": copied,
            "selected_dataset_id": payload["selected_dataset_id"],
            "executed_dataset_ids": payload["executed_dataset_ids"],
            "protocol_dataset_ids": payload["protocol_dataset_ids"],
        },
        message="Search completed test-blind.",
    )


def search_manifest_path(args: argparse.Namespace) -> Path:
    if args.search_manifest:
        return args.search_manifest
    return args.workflow_root / "search" / "training_sweep_manifest.json"


def stage_select_top_k(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    manifest = read_json(require_path(search_manifest_path(args), message="Missing search training_sweep_manifest"))
    records = [dict(row) for row in manifest.get("runs") or [] if isinstance(row, dict)]
    assert_no_test_metrics_for_search(records, stage=SEARCH_STAGE)
    selection = protocol["selection"]
    top_k = protocol["top_k_selection"]
    selected = select_top_configs(
        records,
        metric=str(top_k.get("metric") or selection["metric"]),
        mode=str(selection["mode"]),
        k_per_model=int(top_k["k_per_model"]),
        grouping=str(top_k.get("grouping") or "model_dataset"),
        allow_diagnostic=False,
    )
    output_dir = workflow_root / "selection"
    write_json(output_dir / "selected_configs.json", selected)
    write_csv(output_dir / "selected_configs.csv", list(selected.get("selected_configs") or []))
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        inputs={"search_manifest": str(search_manifest_path(args))},
        outputs={
            "selected_configs_json": str(output_dir / "selected_configs.json"),
            "selected_configs_csv": str(output_dir / "selected_configs.csv"),
            "selected_count": selected.get("selected_count"),
        },
        message="Top-k selected using validation metrics only.",
    )


def stage_generate_final_seeds(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    selected_path = require_path(
        args.selected_configs or workflow_root / "selection" / "selected_configs.json",
        message="Missing selected_configs manifest",
    )
    selected = read_json(selected_path)
    robust_plan = generate_robust_rerun_plan(
        selected,
        final_seeds=[int(seed) for seed in protocol["final_seeds"]],
        stage=ROBUST_VALIDATION_STAGE,
    )
    paths = write_selection_artifacts(workflow_root / "selection", selected, robust_plan)
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        inputs={"selected_configs": str(selected_path)},
        outputs={**paths, "planned_run_count": robust_plan.get("planned_run_count")},
        message="Robust multi-seed rerun plan generated.",
    )


def stage_run_final(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    robust_plan_path = require_path(
        args.robust_rerun_plan or workflow_root / "selection" / "robust_rerun_plan.json",
        message="Missing robust rerun plan",
    )
    robust_plan = read_json(robust_plan_path)
    planned_runs = list(robust_plan.get("planned_runs") or [])
    budget_runs = planned_runs
    if args.dataset_id:
        budget_runs = [row for row in planned_runs if row.get("dataset_id") == args.dataset_id]
    per_model_counts: dict[str, int] = {}
    for row in budget_runs:
        model = str(row.get("model") or "").strip()
        if model:
            per_model_counts[model] = per_model_counts.get(model, 0) + 1
    final_budget_policy = {
        "mode": "equal_n_trials",
        "n_trials_per_model": max(per_model_counts.values(), default=1),
        "source": "robust_rerun_plan",
    }
    plan = {
        "enabled": True,
        "max_runs": int(robust_plan.get("planned_run_count") or len(planned_runs) or 1),
        "error_policy": "continue_on_error",
        "budget_policy": final_budget_policy,
        "search_policy": {"strategy": "selected_final_seeds"},
        "search_plan": {
            "schema": "graph2mat_deeph_final_seed_plan_v1",
            "protocol_id": protocol.get("protocol_id"),
            "protocol_hash": protocol.get("protocol_hash"),
            "planned_run_count": robust_plan.get("planned_run_count"),
            "planned_runs": planned_runs,
        },
        "planned_runs": planned_runs,
    }
    run_id = args.run_id or f"{protocol['protocol_id']}_robust_validation"
    payload = build_runner_payload(
        protocol=protocol,
        plan=plan,
        workflow_root=workflow_root,
        run_id=run_id,
        protocol_stage=ROBUST_VALIDATION_STAGE,
        dry_run=bool(args.dry_run),
        dataset_id=args.dataset_id,
    )
    expected_run_root = workflow_root / "runs" / run_id
    if expected_run_root.exists():
        payload["reuse_run_root"] = True
        payload["resume_training_sweep"] = True
        payload["resume_from_run_root"] = str(expected_run_root)
    selected_plan = payload["_training_sweep_plan"]
    final_dir = workflow_root / "final"
    write_json(final_dir / "run_final_payload.json", payload)
    if args.dry_run:
        manifest = {
            "schema": WORKFLOW_SCHEMA,
            "stage": "run-final",
            "status": "planned_dry_run",
            "run_id": run_id,
            "selected_dataset_id": payload["selected_dataset_id"],
            "executed_dataset_ids": payload["executed_dataset_ids"],
            "protocol_dataset_ids": payload["protocol_dataset_ids"],
            "runner_payload": str(final_dir / "run_final_payload.json"),
            "planned_run_count": len(selected_plan.get("planned_runs") or []),
        }
        write_json(final_dir / "run_final_manifest.json", manifest)
        return stage_manifest(
            workflow_root=workflow_root,
            stage=args.stage,
            status="planned_dry_run",
            outputs={
                "run_final_manifest": str(final_dir / "run_final_manifest.json"),
                "selected_dataset_id": payload["selected_dataset_id"],
                "executed_dataset_ids": payload["executed_dataset_ids"],
                "protocol_dataset_ids": payload["protocol_dataset_ids"],
                "planned_run_count": len(selected_plan.get("planned_runs") or []),
            },
            message="Robust/final runner payload written; no subprocesses launched.",
        )
    runner = Graph2MatDeepHBenchmarkRunner()
    runner.start(payload)
    manifest = wait_for_runner(
        runner=runner,
        output_manifest=final_dir / "run_final_manifest.json",
        poll_seconds=float(args.poll_seconds),
    )
    status = manifest.get("status") if isinstance(manifest.get("status"), dict) else {}
    run_root = Path(str(status.get("run_root") or ""))
    copied = copy_if_exists(run_root / "sweep" / "training_sweep_manifest.json", final_dir / "training_sweep_manifest.json")
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        outputs={
            "run_final_manifest": str(final_dir / "run_final_manifest.json"),
            "final_run_root": str(run_root),
            "training_sweep_manifest": copied,
            "selected_dataset_id": payload["selected_dataset_id"],
            "executed_dataset_ids": payload["executed_dataset_ids"],
            "protocol_dataset_ids": payload["protocol_dataset_ids"],
        },
        message="Robust/final training completed test-blind; final test metrics remain locked.",
    )


def final_run_root(args: argparse.Namespace) -> Path:
    if args.final_run_root:
        return args.final_run_root
    manifest = read_json(args.workflow_root / "stages" / "run-final.json")
    root = ((manifest.get("outputs") or {}).get("final_run_root") or "").strip()
    if not root:
        raise RuntimeError("Missing final run root. Pass --final-run-root or run the run-final stage first.")
    return Path(root)


def final_test_run_root(args: argparse.Namespace) -> Path:
    if args.final_run_root:
        return args.final_run_root
    manifest = read_json(args.workflow_root / "stages" / "run-final-test.json")
    root = ((manifest.get("outputs") or {}).get("final_test_run_root") or "").strip()
    if root:
        return Path(root)
    return final_run_root(args)


def final_evaluation_metric(protocol: dict[str, Any]) -> str:
    section = protocol.get("final_evaluation")
    if not isinstance(section, dict):
        raise RuntimeError("final_evaluation is required for final-test claims.")
    metric = str(section.get("primary_metric") or "").strip()
    if not metric:
        raise RuntimeError("final_evaluation.primary_metric is required for final-test claims.")
    return metric


def final_evaluation_mode(protocol: dict[str, Any]) -> str:
    section = protocol.get("final_evaluation")
    if not isinstance(section, dict):
        raise RuntimeError("final_evaluation is required for final-test claims.")
    mode = str(section.get("mode") or "").strip()
    if not mode:
        raise RuntimeError("final_evaluation.mode is required for final-test claims.")
    return mode


def stage_run_final_test(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    metric = final_evaluation_metric(protocol)
    robust_plan_path = require_path(
        args.robust_rerun_plan or workflow_root / "selection" / "robust_rerun_plan.json",
        message="Missing robust rerun plan",
    )
    robust_plan = read_json(robust_plan_path)
    source_root = final_run_root(args)
    rows = load_rows(source_root)
    output_root = workflow_root / "final_test"
    output_sweep = output_root / "sweep"
    if args.dry_run:
        manifest = {
            "schema": WORKFLOW_SCHEMA,
            "stage": "run-final-test",
            "status": "planned_dry_run",
            "source_final_run_root": str(source_root),
            "robust_rerun_plan": str(robust_plan_path),
            "planned_run_count": len(robust_plan.get("planned_runs") or []),
            "message": "No training or test inference launched.",
        }
        write_json(output_root / "run_final_test_manifest.json", manifest)
        return stage_manifest(
            workflow_root=workflow_root,
            stage=args.stage,
            status="planned_dry_run",
            inputs={"final_run_root": str(source_root), "robust_rerun_plan": str(robust_plan_path)},
            outputs={
                "run_final_test_manifest": str(output_root / "run_final_test_manifest.json"),
                "planned_run_count": len(robust_plan.get("planned_runs") or []),
            },
            message="Final-test materialization planned; no subprocesses launched.",
        )

    normalized_rows: list[dict[str, Any]] = []
    blocked: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or metric_value(row, metric) is None:
            continue
        stage = protocol_stage(row)
        split = metric_split(row)
        if stage == SEARCH_STAGE and split == "test":
            blocked.append(str(row.get("config_id") or row.get("run_id") or index))
            continue
        if stage == FINAL_TEST_STAGE or split == "test":
            item = dict(row)
            item["protocol_stage"] = FINAL_TEST_STAGE
            item["metric_split"] = "test"
            item["final_test_source_run_root"] = str(source_root)
            normalized_rows.append(item)
    if blocked:
        raise RuntimeError(
            "Refusing to promote test metrics produced before explicit final_test stage: "
            + ", ".join(blocked[:10])
        )
    if not normalized_rows:
        manifest = {
            "schema": WORKFLOW_SCHEMA,
            "stage": "run-final-test",
            "status": "missing_final_test_metrics",
            "source_final_run_root": str(source_root),
            "robust_rerun_plan": str(robust_plan_path),
            "planned_run_count": len(robust_plan.get("planned_runs") or []),
            "message": (
                "No final-test metric rows were found. run-final-test does not retrain or invent metrics; "
                "produce locked-test inference/evaluation rows, then rerun this stage."
            ),
        }
        write_json(output_root / "run_final_test_manifest.json", manifest)
        raise RuntimeError("run-final-test found no final_test/test metric rows for the selected final runs.")

    validate_final_evaluation_inputs(
        selected_runs=list(robust_plan.get("planned_runs") or []),
        metric_rows=normalized_rows,
        stage=FINAL_TEST_STAGE,
        metric=metric,
    )
    payload = {
        "schema": "graph2mat_deeph_final_test_metrics_v1",
        "protocol_stage": FINAL_TEST_STAGE,
        "metric_split": "test",
        "source_final_run_root": str(source_root),
        "runs": normalized_rows,
    }
    write_json(output_sweep / "training_sweep_manifest.json", payload)
    write_csv(output_sweep / "training_sweep_metrics.csv", normalized_rows)
    manifest = {
        "schema": WORKFLOW_SCHEMA,
        "stage": "run-final-test",
        "status": "completed",
        "source_final_run_root": str(source_root),
        "final_test_run_root": str(output_root),
        "robust_rerun_plan": str(robust_plan_path),
        "final_test_row_count": len(normalized_rows),
    }
    write_json(output_root / "run_final_test_manifest.json", manifest)
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        inputs={"final_run_root": str(source_root), "robust_rerun_plan": str(robust_plan_path)},
        outputs={
            "run_final_test_manifest": str(output_root / "run_final_test_manifest.json"),
            "final_test_run_root": str(output_root),
            "training_sweep_manifest": str(output_sweep / "training_sweep_manifest.json"),
            "final_test_row_count": len(normalized_rows),
        },
        message="Final-test rows materialized and marked as explicit final_test/test.",
    )


def stage_evaluate_final_test(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    robust_plan_path = require_path(
        args.robust_rerun_plan or workflow_root / "selection" / "robust_rerun_plan.json",
        message="Missing robust rerun plan",
    )
    robust_plan = read_json(robust_plan_path)
    root = final_test_run_root(args)
    metric = final_evaluation_metric(protocol)
    mode = final_evaluation_mode(protocol)
    rows = load_rows(root)
    validate_final_evaluation_inputs(
        selected_runs=list(robust_plan.get("planned_runs") or []),
        metric_rows=rows,
        stage=FINAL_TEST_STAGE,
        metric=metric,
    )
    report = final_statistics_report(
        run_root=root,
        output_dir=workflow_root / "final_test",
        metric=metric,
        mode=mode,
        expected_seeds=[int(seed) for seed in protocol["final_seeds"]],
        min_final_seeds=min(3, len(protocol["final_seeds"])),
    )
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        inputs={"final_run_root": str(root), "robust_rerun_plan": str(robust_plan_path)},
        outputs={"final_statistics_json": report["outputs"]["final_statistics_json"]},
        message="Final test metrics validated and aggregated.",
    )


def stage_generate_report(args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    protocol = load_workflow_protocol(args.protocol, workflow_root)
    root = args.report_run_root or args.final_run_root or final_test_run_root(args)
    metric = final_evaluation_metric(protocol)
    mode = final_evaluation_mode(protocol)
    report = generate_report(
        run_root=root,
        output_dir=workflow_root / "report",
        metric=metric,
        mode=mode,
        compute_threshold=args.compute_threshold,
        final_statistics_path=workflow_root / "final_test" / "final_statistics.json",
        gate_status_path=workflow_root / "gate_status.json",
    )
    evidence = build_evidence_bundle_manifest(
        workflow_root=workflow_root,
        protocol=protocol,
        run_root=root,
        report_outputs=report.get("outputs") if isinstance(report.get("outputs"), dict) else {},
    )
    return stage_manifest(
        workflow_root=workflow_root,
        stage=args.stage,
        status="completed",
        inputs={"run_root": str(root)},
        outputs={**(report.get("outputs") or {}), "evidence_bundle_manifest": evidence["path"]},
        message="Machine-readable final report generated.",
    )


STAGE_HANDLERS = {
    "validate-protocol": stage_validate_protocol,
    "generate-search-plan": stage_generate_search_plan,
    "run-search": stage_run_search,
    "select-top-k": stage_select_top_k,
    "generate-final-seeds": stage_generate_final_seeds,
    "run-final": stage_run_final,
    "run-final-test": stage_run_final_test,
    "evaluate-final-test": stage_evaluate_final_test,
    "generate-report": stage_generate_report,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--workflow-root", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--search-manifest", type=Path, default=None)
    parser.add_argument("--selected-configs", type=Path, default=None)
    parser.add_argument("--robust-rerun-plan", type=Path, default=None)
    parser.add_argument("--final-run-root", type=Path, default=None)
    parser.add_argument("--report-run-root", type=Path, default=None)
    parser.add_argument("--compute-threshold", type=float, default=None)
    parser.add_argument("--verify-datasets", action="store_true")
    parser.add_argument("--write-dataset-manifests", action="store_true")
    return parser.parse_args(argv)


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    args.workflow_root = Path(args.workflow_root)
    args.workflow_root.mkdir(parents=True, exist_ok=True)
    handler = STAGE_HANDLERS[args.stage]
    return handler(args)


def main(argv: list[str] | None = None) -> int:
    manifest = run_stage(parse_args(argv))
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
