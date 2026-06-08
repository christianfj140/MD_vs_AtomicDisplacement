#!/usr/bin/env python3
"""Resumable end-to-end orchestration for Graph2Mat-vs-DeepH.

The script is deliberately a control-plane wrapper: it writes stage
manifests/logs, delegates scientific work to existing repository scripts, and
fails closed when required evidence is missing. It can also summarize already
materialized paper-ready artifacts without retraining or re-running inference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_SCHEMA = "graph2mat_deeph_end_to_end_pipeline_v1"

CLI_EXAMPLES = """examples:
  Dry-run a full workflow:
    python3 Comparison/scripts/g2m_deeph_end_to_end_pipeline.py \\
      --workflow-root Comparison/results/my_workflow \\
      --protocol Comparison/config/my_protocol.json \\
      --dataset-root Comparison/datasets/my_dataset \\
      --strict --dry-run

  Resume from DeepH equivalence with explicit runnable inputs:
    python3 Comparison/scripts/g2m_deeph_end_to_end_pipeline.py \\
      --workflow-root Comparison/results/my_workflow \\
      --from-stage equivalence --resume \\
      --frozen-split-manifest Comparison/datasets/my_dataset/frozen_split_manifest.json \\
      --graph2mat-result-dir Comparison/datasets/my_dataset/reference_input \\
      --deeph-run-glob 'runs/*/sweep/deeph/*/*' \\
      --equivalence-jobs 4 --strict

  Produce a paper-ready strict summary from an existing final workflow root:
    python3 Comparison/scripts/g2m_deeph_end_to_end_pipeline.py \\
      --workflow-root Comparison/results/g2m_deeph_iid600_phaseB_intermediate_spectral_refine_v1 \\
      --stages equivalence,summary
"""

STAGE_ORDER = (
    "generate-dataset",
    "verify-dataset",
    "validate-protocol",
    "generate-search-plan",
    "run-search",
    "select-top-k",
    "generate-final-seeds",
    "run-final",
    "materialize-final-test",
    "run-final-test",
    "equivalence",
    "final-stats",
    "gate-check",
    "report",
    "release-manifest",
    "summary",
)

STAGE_ALIASES = {
    "evaluate-final-test": "final-stats",
    "generate-report": "report",
    "materialize-test": "materialize-final-test",
}

ALL_STAGE_NAMES = tuple(STAGE_ORDER) + tuple(STAGE_ALIASES)

FINAL_WORKFLOW_STAGE_MAP = {
    "validate-protocol": "validate-protocol",
    "generate-search-plan": "generate-search-plan",
    "run-search": "run-search",
    "select-top-k": "select-top-k",
    "generate-final-seeds": "generate-final-seeds",
    "run-final": "run-final",
    "run-final-test": "run-final-test",
    "final-stats": "evaluate-final-test",
    "report": "generate-report",
}

FINAL_WORKFLOW_STAGES = {
    *FINAL_WORKFLOW_STAGE_MAP,
}

HEAVY_STAGES = {
    "generate-dataset",
    "run-search",
    "run-final",
    "materialize-final-test",
}


@dataclass(frozen=True)
class CommandPlan:
    command: list[str]
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    message: str = ""
    heavy: bool = False
    internal: bool = False


@dataclass(frozen=True)
class InternalStageResult:
    status: str
    returncode: int | None = 0
    stdout: str = ""
    stderr: str = ""
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    message: str = ""


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


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


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def canonical_stage_name(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def stage_manifest_path(workflow_root: Path, stage: str) -> Path:
    return workflow_root / "stages" / f"{stage}.json"


def stage_log_paths(workflow_root: Path, stage: str) -> tuple[Path, Path]:
    logs_dir = workflow_root / "logs"
    return logs_dir / f"{stage}.stdout.log", logs_dir / f"{stage}.stderr.log"


def pipeline_state_path(workflow_root: Path) -> Path:
    return workflow_root / "pipeline_state.json"


def append_state(workflow_root: Path, stage_manifest: dict[str, Any], *, overall_status: str) -> dict[str, Any]:
    state_path = pipeline_state_path(workflow_root)
    state = read_json(state_path)
    history = state.get("history") if isinstance(state.get("history"), list) else []
    history.append(
        {
            "stage": stage_manifest.get("stage"),
            "status": stage_manifest.get("status"),
            "started_at": stage_manifest.get("started_at"),
            "finished_at": stage_manifest.get("finished_at"),
            "returncode": stage_manifest.get("returncode"),
        }
    )
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    stages[str(stage_manifest.get("stage") or "")] = stage_manifest
    completed = [
        name
        for name, manifest in stages.items()
        if isinstance(manifest, dict) and manifest.get("status") == "completed"
    ]
    payload = {
        "schema": SCRIPT_SCHEMA,
        "updated_at": timestamp(),
        "workflow_root": str(workflow_root),
        "status": overall_status,
        "last_stage": stage_manifest.get("stage"),
        "completed_stages": sorted(
            completed,
            key=lambda item: STAGE_ORDER.index(canonical_stage_name(item))
            if canonical_stage_name(item) in STAGE_ORDER
            else 999,
        ),
        "stages": stages,
        "history": history,
    }
    write_json(state_path, payload)
    return payload


def write_stage_manifest(
    *,
    workflow_root: Path,
    stage: str,
    status: str,
    plan: CommandPlan,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    returncode: int | None,
    stdout_log: Path,
    stderr_log: Path,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    message: str = "",
) -> dict[str, Any]:
    manifest = {
        "schema": SCRIPT_SCHEMA,
        "stage": stage,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "workflow_root": str(workflow_root),
        "cwd": str(REPO_ROOT),
        "command": plan.command,
        "returncode": returncode,
        "inputs": plan.inputs,
        "outputs": {
            **plan.outputs,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        },
        "blockers": blockers or [],
        "warnings": warnings or [],
        "message": message or plan.message,
        "heavy": plan.heavy,
        "internal": plan.internal,
    }
    write_json(stage_manifest_path(workflow_root, stage), manifest)
    return manifest


def ensure_log_files(stdout_log: Path, stderr_log: Path, *, stdout: str = "", stderr: str = "") -> None:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")


def canonical_protocol_path(workflow_root: Path) -> Path:
    return workflow_root / "protocol" / "validated_protocol.json"


def python_executable(args: argparse.Namespace) -> str:
    return str(args.python_executable or sys.executable)


def script_path(name: str) -> str:
    return str(REPO_ROOT / "Comparison" / "scripts" / name)


def optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value not in (None, ""):
        command.extend([flag, str(value)])


def optional_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def build_final_workflow_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    workflow_stage = FINAL_WORKFLOW_STAGE_MAP[stage]
    command = [
        python_executable(args),
        script_path("g2m_deeph_final_workflow.py"),
        "--stage",
        workflow_stage,
        "--workflow-root",
        str(args.workflow_root),
    ]
    optional_arg(command, "--protocol", args.protocol)
    optional_arg(command, "--run-id", args.run_id)
    optional_arg(command, "--dataset-id", args.dataset_id)
    optional_arg(command, "--poll-seconds", args.poll_seconds)
    optional_arg(command, "--search-manifest", args.search_manifest)
    optional_arg(command, "--selected-configs", args.selected_configs)
    optional_arg(command, "--robust-rerun-plan", args.robust_rerun_plan)
    optional_arg(command, "--final-run-root", args.final_run_root)
    optional_arg(command, "--report-run-root", args.report_run_root)
    optional_arg(command, "--compute-threshold", args.compute_threshold)
    optional_flag(command, "--verify-datasets", bool(args.verify_datasets))
    optional_flag(command, "--write-dataset-manifests", bool(args.write_dataset_manifests))
    optional_flag(command, "--dry-run", bool(args.dry_run))
    return CommandPlan(
        command=command,
        inputs={
            "protocol": str(args.protocol) if args.protocol else "",
            "workflow_root": str(args.workflow_root),
            "dataset_id": args.dataset_id,
            "workflow_stage": workflow_stage,
        },
        outputs={"stage_manifest": str(stage_manifest_path(args.workflow_root, stage))},
        message=f"Delegated to g2m_deeph_final_workflow.py --stage {workflow_stage}.",
        heavy=workflow_stage in HEAVY_STAGES,
    )


def build_generate_dataset_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    command = [python_executable(args), str(REPO_ROOT / "MD" / "scripts" / "generate_md_dataset.py")]
    env: dict[str, str] = {}
    inputs: dict[str, Any] = {"pipeline_config": ""}
    if args.pipeline_config:
        env["PIPELINE_CONFIG_PATH"] = str(args.pipeline_config)
        inputs["pipeline_config"] = str(args.pipeline_config)
    return CommandPlan(
        command=command,
        env=env,
        inputs=inputs,
        outputs={"stage_manifest": str(stage_manifest_path(args.workflow_root, stage))},
        message="Delegated to MD/scripts/generate_md_dataset.py.",
        heavy=True,
    )


def build_verify_dataset_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    if not args.protocol:
        raise RuntimeError("verify-dataset requires --protocol.")
    output = args.workflow_root / "dataset_verification.json"
    command = [
        python_executable(args),
        script_path("g2m_deeph_verify_protocol_datasets.py"),
        "--protocol",
        str(args.protocol),
        "--output",
        str(output),
    ]
    optional_flag(command, "--strict", bool(args.strict))
    optional_flag(command, "--write-manifests", bool(args.write_dataset_manifests))
    return CommandPlan(
        command=command,
        inputs={"protocol": str(args.protocol)},
        outputs={"dataset_verification": str(output)},
        message="Delegated to g2m_deeph_verify_protocol_datasets.py.",
    )


def gate_protocol_path(args: argparse.Namespace) -> Path:
    if args.protocol:
        return Path(args.protocol)
    return canonical_protocol_path(args.workflow_root)


def build_gate_check_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    protocol = gate_protocol_path(args)
    command = [
        python_executable(args),
        script_path("g2m_deeph_gate_check.py"),
        "--protocol",
        str(protocol),
        "--workflow-root",
        str(args.workflow_root),
        "--output",
        str(args.workflow_root / "gate_status.json"),
    ]
    optional_arg(command, "--run-root", args.final_run_root or args.report_run_root)
    return CommandPlan(
        command=command,
        inputs={"protocol": str(protocol), "workflow_root": str(args.workflow_root)},
        outputs={"gate_status": str(args.workflow_root / "gate_status.json")},
        message="Delegated to g2m_deeph_gate_check.py.",
    )


def build_release_manifest_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    if not args.dataset_root:
        raise RuntimeError("release-manifest requires --dataset-root.")
    output = args.workflow_root / "release_manifest.json"
    command = [
        python_executable(args),
        script_path("g2m_deeph_release_manifest.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--workflow-root",
        str(args.workflow_root),
        "--output",
        str(output),
    ]
    optional_arg(command, "--run-root", args.final_run_root or args.report_run_root)
    optional_flag(command, "--strict", bool(args.strict))
    return CommandPlan(
        command=command,
        inputs={
            "dataset_root": str(args.dataset_root),
            "workflow_root": str(args.workflow_root),
        },
        outputs={"release_manifest": str(output)},
        message="Delegated to g2m_deeph_release_manifest.py.",
    )


def build_materialize_final_test_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    if not args.final_run_root:
        raise RuntimeError("materialize-final-test requires --final-run-root.")
    if not args.confirm_final_test_open:
        raise RuntimeError("materialize-final-test requires --confirm-final-test-open.")
    command = [
        python_executable(args),
        script_path("g2m_deeph_materialize_validation_metrics.py"),
        "--run-root",
        str(args.final_run_root),
        "--model",
        str(args.materialize_model),
        "--jobs",
        str(args.materialize_jobs),
        "--split",
        "test",
        "--protocol-stage",
        "final_test",
        "--confirm-final-test-open",
    ]
    optional_arg(command, "--limit", args.materialize_limit)
    optional_flag(command, "--overwrite", bool(args.materialize_overwrite))
    optional_flag(command, "--watch", bool(args.materialize_watch))
    optional_flag(command, "--skip-existing", bool(args.materialize_skip_existing))
    optional_arg(command, "--max-cycles", args.materialize_max_cycles)
    optional_arg(command, "--poll-seconds", args.poll_seconds)
    return CommandPlan(
        command=command,
        inputs={
            "final_run_root": str(args.final_run_root),
            "split": "test",
            "protocol_stage": "final_test",
            "model": args.materialize_model,
            "jobs": args.materialize_jobs,
        },
        outputs={"stage_manifest": str(stage_manifest_path(args.workflow_root, stage))},
        message="Delegated to g2m_deeph_materialize_validation_metrics.py for locked final-test metrics.",
        heavy=True,
    )


def _candidate_deeph_run_root(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    if (path / "deeph" / "inference").is_dir() or (path / "inference").is_dir():
        return path
    if path.name == "inference" and path.parent.name == "deeph":
        return path.parent.parent
    return None


def discover_deeph_run_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    for raw in args.deeph_run_root or []:
        roots.append(Path(raw))
    for pattern in args.deeph_run_glob or []:
        roots.extend(Path(path) for path in sorted(args.workflow_root.glob(str(pattern))))
    if args.final_run_root and not roots:
        final_root = Path(args.final_run_root)
        candidates = [
            path
            for path in final_root.rglob("*")
            if path.is_dir() and ((path / "deeph" / "inference").is_dir() or (path / "inference").is_dir())
        ]
        roots.extend(candidates)
    if not roots:
        for base in (
            args.workflow_root / "final_test",
            args.workflow_root / "runs",
            args.workflow_root,
        ):
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                candidate = _candidate_deeph_run_root(path)
                if candidate is not None:
                    roots.append(candidate)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidate = _candidate_deeph_run_root(Path(root)) or Path(root)
        resolved_key = str(candidate)
        if resolved_key not in seen:
            seen.add(resolved_key)
            deduped.append(candidate)
    return deduped


def deeph_inference_dir(run_root: Path) -> Path:
    run_root = Path(run_root)
    nested = run_root / "deeph" / "inference"
    if nested.is_dir():
        return nested
    direct = run_root / "inference"
    if direct.is_dir():
        return direct
    return run_root


def build_equivalence_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    output_dir = args.equivalence_output_dir or args.workflow_root / "equivalence_strict"
    run_roots: list[Path] = [Path(raw) for raw in args.deeph_run_root or []]
    for pattern in args.deeph_run_glob or []:
        run_roots.extend(Path(path) for path in sorted(args.workflow_root.glob(str(pattern))))
    return CommandPlan(
        command=[
            "internal-parallel",
            "deeph_raw_global_equivalence_preflight.py",
            "--jobs",
            str(args.equivalence_jobs),
        ],
        inputs={
            "frozen_split_manifest": str(args.frozen_split_manifest) if args.frozen_split_manifest else "",
            "graph2mat_result_dir": str(args.graph2mat_result_dir) if args.graph2mat_result_dir else "",
            "deeph_run_roots": [str(path) for path in run_roots],
            "sample_limit": args.equivalence_sample_limit,
            "sample_ids": list(args.equivalence_sample_id or []),
        },
        outputs={
            "equivalence_output_dir": str(output_dir),
            "equivalence_summary": str(output_dir / "equivalence_strict_summary.json"),
        },
        message="Runs DeepH raw/global equivalence preflight in parallel and refreshes adapter manifests.",
        internal=True,
    )


def build_summary_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    output = args.summary_output or args.workflow_root / "strict_summary.json"
    return CommandPlan(
        command=[],
        inputs={"workflow_root": str(args.workflow_root)},
        outputs={
            "pipeline_state": str(pipeline_state_path(args.workflow_root)),
            "strict_summary": str(output),
        },
        message="Internal strict summary stage.",
        internal=True,
    )


def build_stage_plan(stage: str, args: argparse.Namespace) -> CommandPlan:
    if stage in FINAL_WORKFLOW_STAGES:
        return build_final_workflow_plan(stage, args)
    if stage == "generate-dataset":
        return build_generate_dataset_plan(stage, args)
    if stage == "verify-dataset":
        return build_verify_dataset_plan(stage, args)
    if stage == "equivalence":
        return build_equivalence_plan(stage, args)
    if stage == "gate-check":
        return build_gate_check_plan(stage, args)
    if stage == "release-manifest":
        return build_release_manifest_plan(stage, args)
    if stage == "materialize-final-test":
        return build_materialize_final_test_plan(stage, args)
    if stage == "summary":
        return build_summary_plan(stage, args)
    raise RuntimeError(f"Unsupported stage: {stage}")


def _sanitize_run_id(path: Path, index: int) -> str:
    name = path.name or f"deeph_run_{index:03d}"
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def _equivalence_command(args: argparse.Namespace, run_root: Path, output_dir: Path) -> list[str]:
    inference_dir = deeph_inference_dir(run_root)
    command = [
        python_executable(args),
        script_path("deeph_raw_global_equivalence_preflight.py"),
        "--frozen-split-manifest",
        str(args.frozen_split_manifest),
        "--graph2mat-result-dir",
        str(args.graph2mat_result_dir),
        "--deeph-processed-dir",
        str(inference_dir),
        "--deeph-predictions-dir",
        str(inference_dir),
        "--sample-limit",
        str(args.equivalence_sample_limit),
        "--output-dir",
        str(output_dir),
    ]
    for sample_id in args.equivalence_sample_id or []:
        command.extend(["--sample-id", str(sample_id)])
    optional_arg(command, "--matrix-tolerance", args.matrix_tolerance)
    optional_arg(command, "--eigenvalue-tolerance", args.eigenvalue_tolerance)
    return command


def _sample_dirs_for_adapter_refresh(inference_dir: Path) -> list[Path]:
    if not inference_dir.exists():
        return []
    candidates = [
        path
        for path in inference_dir.rglob("*")
        if path.is_dir()
        and (
            (path / "hamiltonians_pred.h5").exists()
            or (path / "rh_pred.h5").exists()
            or (path / "raw_global_equivalence_evidence.json").exists()
        )
    ]
    return sorted(candidates)


def refresh_adapter_manifest(run_root: Path) -> dict[str, Any]:
    scripts_dir = REPO_ROOT / "Comparison" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from deeph_prediction_adapter import (  # noqa: WPS433 - local import keeps skeleton import-light.
        DeepHPredictionAdapterError,
        adapt_deeph_prediction_sample,
        write_adapter_manifest,
    )

    inference_dir = deeph_inference_dir(run_root)
    results = []
    errors: list[str] = []
    for sample_dir in _sample_dirs_for_adapter_refresh(inference_dir):
        try:
            results.append(
                adapt_deeph_prediction_sample(
                    work_dir=sample_dir,
                    processed_sample_dir=sample_dir,
                    sample_id=sample_dir.name,
                )
            )
        except DeepHPredictionAdapterError as exc:
            errors.append(f"{sample_dir}: {exc}")
    if not results:
        return {
            "status": "failed",
            "adapter_manifest": "",
            "sample_count": 0,
            "blockers": errors or [f"No DeepH prediction/evidence sample directories found under {inference_dir}."],
        }
    manifest_path = inference_dir / "adapter_manifest.json"
    manifest = write_adapter_manifest(manifest_path, results)
    status = "completed" if not errors and manifest.get("equivalence_gate", {}).get("robust_claim_allowed") is True else "failed"
    return {
        "status": status,
        "adapter_manifest": str(manifest_path),
        "sample_count": len(results),
        "raw_global_equivalence_proven_count": manifest.get("raw_global_equivalence_proven_count"),
        "robust_matrix_metrics_allowed": manifest.get("robust_matrix_metrics_allowed"),
        "blockers": errors,
    }


def _run_one_equivalence_job(args: argparse.Namespace, run_root: Path, output_base: Path, index: int) -> dict[str, Any]:
    run_id = _sanitize_run_id(run_root, index)
    run_output = output_base / run_id
    log_dir = output_base / "logs"
    stdout_log = log_dir / f"{run_id}.stdout.log"
    stderr_log = log_dir / f"{run_id}.stderr.log"
    command = _equivalence_command(args, run_root, run_output)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    ensure_log_files(stdout_log, stderr_log, stdout=completed.stdout, stderr=completed.stderr)
    adapter_refresh: dict[str, Any] = {"status": "not_run"}
    blockers: list[str] = []
    if completed.returncode != 0:
        blockers.append(f"equivalence_preflight_failed:{run_id}:returncode={completed.returncode}")
    else:
        adapter_refresh = refresh_adapter_manifest(run_root)
        if adapter_refresh.get("status") != "completed":
            blockers.extend(str(item) for item in adapter_refresh.get("blockers") or [])
            if not blockers:
                blockers.append(f"adapter_manifest_refresh_failed:{run_id}")
    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "inference_dir": str(deeph_inference_dir(run_root)),
        "status": "completed" if not blockers else "failed",
        "returncode": completed.returncode,
        "command": command,
        "output_dir": str(run_output),
        "preflight_output": str(run_output / "deeph_raw_global_equivalence_preflight.json"),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "adapter_refresh": adapter_refresh,
        "blockers": blockers,
    }


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_is_present(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _preflight_sample_counts(payload: dict[str, Any]) -> tuple[int, int, int]:
    samples = payload.get("samples") if isinstance(payload.get("samples"), list) else []
    seen = _int_value(payload.get("samples_seen"), len(samples))
    proven = _int_value(
        payload.get("samples_proven"),
        sum(1 for item in samples if isinstance(item, dict) and item.get("status") == "proven"),
    )
    failed = _int_value(
        payload.get("samples_failed"),
        sum(1 for item in samples if isinstance(item, dict) and item.get("status") not in (None, "proven")),
    )
    return seen, proven, failed


def summarize_existing_equivalence(output_base: Path) -> dict[str, Any]:
    preflight_paths = sorted(output_base.glob("*/deeph_raw_global_equivalence_preflight.json"))
    jobs: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, preflight_path in enumerate(preflight_paths, start=1):
        payload = read_json(preflight_path)
        run_id = preflight_path.parent.name or f"deeph_run_{index:03d}"
        status = str(payload.get("status") or "missing")
        samples_seen, samples_proven, samples_failed = _preflight_sample_counts(payload)
        job_blockers = [str(item) for item in payload.get("blockers") or [] if item]
        if status != "proven":
            job_blockers.append(f"preflight_status:{status}")
        if samples_seen <= 0:
            job_blockers.append("no_equivalence_samples_seen")
        if samples_failed > 0:
            job_blockers.append(f"samples_failed:{samples_failed}")
        if samples_seen > 0 and samples_proven < samples_seen:
            job_blockers.append(f"samples_not_proven:{samples_seen - samples_proven}")
        job_status = "completed" if not job_blockers else "failed"
        blockers.extend(f"{run_id}:{item}" for item in job_blockers)
        jobs.append(
            {
                "run_id": run_id,
                "status": job_status,
                "preflight_status": status,
                "preflight_output": str(preflight_path),
                "samples_seen": samples_seen,
                "samples_proven": samples_proven,
                "samples_failed": samples_failed,
                "output_dir": str(preflight_path.parent),
                "blockers": job_blockers,
            }
        )
    failed_jobs = [job for job in jobs if job.get("status") != "completed"]
    if not jobs:
        status = "missing"
        blockers.append(f"missing_equivalence_preflight_outputs:{output_base}")
    else:
        status = "completed" if not failed_jobs else "failed"
    return {
        "schema": SCRIPT_SCHEMA,
        "stage": "equivalence",
        "status": status,
        "source": "existing_preflight_outputs",
        "generated_at": timestamp(),
        "jobs": jobs,
        "run_count": len(jobs),
        "completed_count": len(jobs) - len(failed_jobs),
        "failed_count": len(failed_jobs),
        "samples_seen_total": sum(_int_value(job.get("samples_seen")) for job in jobs),
        "samples_proven_total": sum(_int_value(job.get("samples_proven")) for job in jobs),
        "samples_failed_total": sum(_int_value(job.get("samples_failed")) for job in jobs),
        "blockers": blockers,
        "output_dir": str(output_base),
    }


def write_existing_equivalence_summary(output_base: Path) -> dict[str, Any]:
    summary = summarize_existing_equivalence(output_base)
    write_json(output_base / "equivalence_strict_summary.json", summary)
    return summary


def run_equivalence_stage(args: argparse.Namespace) -> InternalStageResult:
    output_base = args.equivalence_output_dir or args.workflow_root / "equivalence_strict"
    output_base.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    if not args.frozen_split_manifest:
        blockers.append("equivalence requires --frozen-split-manifest.")
    if not args.graph2mat_result_dir:
        blockers.append("equivalence requires --graph2mat-result-dir.")
    if blockers:
        existing_summary = summarize_existing_equivalence(output_base)
        if existing_summary.get("run_count"):
            write_json(output_base / "equivalence_strict_summary.json", existing_summary)
            status = str(existing_summary.get("status") or "failed")
            return InternalStageResult(
                status=status,
                returncode=0 if status == "completed" else 1,
                stdout=json.dumps(json_safe(existing_summary), indent=2, sort_keys=True) + "\n",
                stderr="\n".join(str(item) for item in existing_summary.get("blockers") or []) + "\n",
                blockers=[str(item) for item in existing_summary.get("blockers") or []],
                warnings=[
                    "equivalence stage reused already materialized strict preflight evidence "
                    "because runnable preflight inputs were incomplete."
                ],
                outputs={
                    "equivalence_summary": str(output_base / "equivalence_strict_summary.json"),
                    "equivalence_output_dir": str(output_base),
                },
                message="Synthesized DeepH equivalence summary from existing preflight evidence.",
            )
    run_roots = discover_deeph_run_roots(args)
    if not run_roots:
        blockers.append("equivalence requires at least one --deeph-run-root, --deeph-run-glob match, or discoverable --final-run-root.")
    if blockers:
        existing_summary = summarize_existing_equivalence(output_base)
        if existing_summary.get("run_count"):
            write_json(output_base / "equivalence_strict_summary.json", existing_summary)
            status = str(existing_summary.get("status") or "failed")
            return InternalStageResult(
                status=status,
                returncode=0 if status == "completed" else 1,
                stdout=json.dumps(json_safe(existing_summary), indent=2, sort_keys=True) + "\n",
                stderr="\n".join(str(item) for item in existing_summary.get("blockers") or []) + "\n",
                blockers=[str(item) for item in existing_summary.get("blockers") or []],
                warnings=[
                    "equivalence stage reused already materialized strict preflight evidence "
                    "because runnable preflight inputs were incomplete."
                ],
                outputs={
                    "equivalence_summary": str(output_base / "equivalence_strict_summary.json"),
                    "equivalence_output_dir": str(output_base),
                },
                message="Synthesized DeepH equivalence summary from existing preflight evidence.",
            )
        summary = {
            "schema": SCRIPT_SCHEMA,
            "stage": "equivalence",
            "status": "blocked",
            "run_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "blockers": blockers,
        }
        write_json(output_base / "equivalence_strict_summary.json", summary)
        return InternalStageResult(
            status="blocked",
            returncode=None,
            stdout=json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
            stderr="\n".join(blockers) + "\n",
            blockers=blockers,
            outputs={"equivalence_summary": str(output_base / "equivalence_strict_summary.json")},
            message="DeepH equivalence could not start because required inputs are missing.",
        )

    max_workers = max(1, int(args.equivalence_jobs or 1))
    jobs: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_one_equivalence_job, args, run_root, output_base, index)
            for index, run_root in enumerate(run_roots, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            jobs.append(future.result())
    jobs = sorted(jobs, key=lambda item: str(item.get("run_id") or ""))
    failed = [job for job in jobs if job.get("status") != "completed"]
    summary = {
        "schema": SCRIPT_SCHEMA,
        "stage": "equivalence",
        "status": "completed" if not failed else "failed",
        "jobs": jobs,
        "run_count": len(jobs),
        "completed_count": len(jobs) - len(failed),
        "failed_count": len(failed),
        "blockers": [blocker for job in failed for blocker in job.get("blockers", [])],
        "output_dir": str(output_base),
    }
    summary_path = output_base / "equivalence_strict_summary.json"
    write_json(summary_path, summary)
    return InternalStageResult(
        status=str(summary["status"]),
        returncode=0 if not failed else 1,
        stdout=json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        stderr="\n".join(str(item) for item in summary["blockers"]) + ("\n" if summary["blockers"] else ""),
        blockers=[str(item) for item in summary["blockers"]],
        outputs={"equivalence_summary": str(summary_path), "equivalence_output_dir": str(output_base)},
        message="DeepH equivalence preflight and adapter manifest refresh completed."
        if not failed
        else "At least one DeepH equivalence job failed.",
    )


def _extend_blockers(blockers: list[str], prefix: str, value: Any) -> None:
    if not value:
        return
    if isinstance(value, list):
        blockers.extend(f"{prefix}:{item}" for item in value if item)
    elif isinstance(value, dict):
        blockers.extend(f"{prefix}:{key}={item}" for key, item in value.items() if item)
    else:
        blockers.append(f"{prefix}:{value}")


def _final_seed_rows(final_stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows = final_stats.get("final_seed_summary")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _telemetry_blockers(final_stats: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    required_fields = ("gpu_hours_mean", "peak_gpu_memory_mb_mean")
    optional_fields = ("gpu_hours_std", "peak_gpu_memory_mb_std", "inference_seconds_mean")
    for index, row in enumerate(_final_seed_rows(final_stats), start=1):
        label = "/".join(
            str(row.get(key) or "")
            for key in ("model", "dataset_id", "selected_config_id")
            if row.get(key)
        ) or f"final_seed_summary_row_{index}"
        for field_name in required_fields:
            if not _float_is_present(row.get(field_name)):
                blockers.append(f"missing_telemetry_field:{label}:{field_name}")
        if not any(_float_is_present(row.get(field_name)) for field_name in optional_fields):
            blockers.append(f"missing_telemetry_detail:{label}:std_or_inference_seconds")
    if final_stats and not _final_seed_rows(final_stats):
        blockers.append("missing_final_seed_summary")
    return blockers


def _winner_by_dataset(winner_decision: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_decisions = winner_decision.get("dataset_decisions")
    if not isinstance(dataset_decisions, list):
        return []
    winners: list[dict[str, Any]] = []
    for decision in dataset_decisions:
        if not isinstance(decision, dict):
            continue
        winners.append(
            {
                "dataset_id": decision.get("dataset_id"),
                "precision_winner": decision.get("precision_winner"),
                "winner_config_id": decision.get("winner_config_id"),
                "runner_up_model": decision.get("runner_up_model"),
                "runner_up_config_id": decision.get("runner_up_config_id"),
                "effect_size_best_vs_second": decision.get("effect_size_best_vs_second"),
                "ci_rule_passed": decision.get("ci_rule_passed"),
                "gates_failed": decision.get("gates_failed") or [],
                "best_config_by_model": decision.get("best_config_by_model") or {},
            }
        )
    return winners


def _global_winner_from_dataset_winners(winners: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [
        str(item.get("precision_winner"))
        for item in winners
        if item.get("precision_winner")
    ]
    if not labels:
        return {"winner": None, "reason": "no_dataset_winners"}
    unique = sorted(set(labels))
    if len(unique) == 1 and len(labels) == len(winners):
        return {"winner": unique[0], "reason": "all_dataset_winners_agree", "dataset_count": len(labels)}
    return {
        "winner": None,
        "reason": "dataset_winners_disagree",
        "dataset_winners": labels,
        "unique_winners": unique,
    }


def _write_strict_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Graph2Mat vs DeepH Strict Summary",
        "",
        f"- Workflow root: `{payload.get('workflow_root')}`",
        f"- Robust claim allowed: `{payload.get('robust_claim_allowed')}`",
        f"- Claim status: `{payload.get('claim_status')}`",
        f"- Precision winner: `{payload.get('precision_winner')}`",
        f"- Diagnostic precision winner: `{payload.get('diagnostic_precision_winner')}`",
        f"- Global winner: `{(payload.get('global_winner') or {}).get('winner')}`",
        "",
        "## Winners By Dataset",
    ]
    winners = payload.get("winner_by_dataset") if isinstance(payload.get("winner_by_dataset"), list) else []
    if winners:
        for winner in winners:
            lines.append(
                "- "
                f"`{winner.get('dataset_id')}`: {winner.get('precision_winner')} "
                f"({winner.get('winner_config_id')}) vs "
                f"{winner.get('runner_up_model')} ({winner.get('runner_up_config_id')})"
            )
    else:
        lines.append("- No dataset-level winner decision available.")
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    lines.extend(["", "## Blockers"])
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None.")
    next_actions = payload.get("next_actions") if isinstance(payload.get("next_actions"), list) else []
    lines.extend(["", "## Next Actions"])
    if next_actions:
        lines.extend(f"- {action}" for action in next_actions)
    else:
        lines.append("- No action required for the strict evidence currently checked.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _next_actions_from_blockers(blockers: list[str]) -> list[str]:
    actions: list[str] = []
    joined = "\n".join(blockers)
    if "missing_equivalence" in joined or "equivalence" in joined:
        actions.append("Run or repair DeepH raw/global equivalence and refresh adapter manifests.")
    if "telemetry" in joined:
        actions.append("Regenerate final statistics from rows that include raw telemetry fields.")
    if "release_manifest" in joined or "release" in joined:
        actions.append("Regenerate `g2m_deeph_release_manifest.py --strict` and materialize missing artifacts.")
    if "final_statistics" in joined:
        actions.append("Run `final-stats` after final-test rows are complete.")
    if "gate" in joined:
        actions.append("Run `gate-check` and inspect its blockers.")
    return sorted(set(actions))


def run_summary_stage(args: argparse.Namespace) -> InternalStageResult:
    output = args.summary_output or args.workflow_root / "strict_summary.json"
    final_stats_path = args.workflow_root / "final_test" / "final_statistics.json"
    gate_path = args.workflow_root / "gate_status.json"
    release_path = args.workflow_root / "release_manifest.json"
    report_path = args.workflow_root / "report" / "final_report.json"
    equivalence_dir = args.equivalence_output_dir or args.workflow_root / "equivalence_strict"
    equivalence_path = equivalence_dir / "equivalence_strict_summary.json"

    final_stats = read_json(final_stats_path)
    gate = read_json(gate_path)
    release = read_json(release_path)
    report = read_json(report_path)
    equivalence = read_json(equivalence_path)
    if not equivalence:
        existing_equivalence = summarize_existing_equivalence(equivalence_dir)
        if existing_equivalence.get("run_count"):
            write_json(equivalence_path, existing_equivalence)
            equivalence = existing_equivalence
    winner_decision = final_stats.get("winner_decision") if isinstance(final_stats.get("winner_decision"), dict) else {}
    winner_by_dataset = _winner_by_dataset(winner_decision)
    global_winner = _global_winner_from_dataset_winners(winner_by_dataset)

    blockers: list[str] = []
    if not final_stats:
        blockers.append(f"missing_final_statistics:{final_stats_path}")
    if not gate:
        blockers.append(f"missing_gate_status:{gate_path}")
    if not release:
        blockers.append(f"missing_release_manifest:{release_path}")
    if not report:
        blockers.append(f"missing_final_report:{report_path}")
    if not equivalence:
        blockers.append(f"missing_equivalence_strict_summary:{equivalence_path}")
    if final_stats and winner_decision.get("robust_claim_allowed") is not True:
        blockers.append("final_statistics_robust_claim_allowed:false")
    if gate and gate.get("robust_claim_allowed") is not True:
        blockers.append(f"gate_robust_claim_allowed:false:{gate.get('claim_status') or 'unknown'}")
    if release and release.get("status") != "complete":
        blockers.append(f"release_manifest_status:{release.get('status') or 'unknown'}")
    if equivalence and equivalence.get("status") != "completed":
        blockers.append(f"equivalence_status:{equivalence.get('status') or 'unknown'}")
    if final_stats:
        blockers.extend(_telemetry_blockers(final_stats))
    if release and release.get("missing_required"):
        blockers.append(f"release_manifest_missing_required_count:{len(release.get('missing_required') or [])}")
    if release and release.get("forbidden_reference_findings"):
        blockers.append(
            f"release_manifest_forbidden_reference_count:{len(release.get('forbidden_reference_findings') or [])}"
        )
    _extend_blockers(blockers, "final_statistics", winner_decision.get("gates_failed"))
    _extend_blockers(blockers, "gate_check", gate.get("blockers") or [])
    _extend_blockers(blockers, "release_manifest_missing_required", release.get("missing_required") or [])
    _extend_blockers(blockers, "release_manifest_forbidden_reference", release.get("forbidden_reference_findings") or [])
    _extend_blockers(blockers, "equivalence", equivalence.get("blockers") or [])

    stats_allowed = bool(final_stats) and winner_decision.get("robust_claim_allowed") is True
    gate_allowed = bool(gate) and gate.get("robust_claim_allowed") is True
    release_complete = bool(release) and release.get("status") == "complete" and not release.get("missing_required") and not release.get("forbidden_reference_findings")
    equivalence_complete = bool(equivalence) and equivalence.get("status") == "completed" and not equivalence.get("blockers")
    global_winner_agrees = global_winner.get("winner") is not None
    if stats_allowed and winner_by_dataset and not global_winner_agrees:
        blockers.append("global_winner_blocked:dataset_winners_disagree")
    robust_claim_allowed = bool(
        stats_allowed
        and gate_allowed
        and release_complete
        and equivalence_complete
        and global_winner_agrees
        and not blockers
    )
    unique_blockers = sorted(set(str(item) for item in blockers if item))
    md_output = output.with_suffix(".md")
    payload = {
        "schema": SCRIPT_SCHEMA,
        "stage": "summary",
        "status": "completed" if robust_claim_allowed else "blocked",
        "workflow_root": str(args.workflow_root),
        "generated_at": timestamp(),
        "robust_claim_allowed": robust_claim_allowed,
        "gate_robust_claim_allowed": gate_allowed,
        "final_statistics_robust_claim_allowed": stats_allowed,
        "strict_release_manifest_complete": release_complete,
        "equivalence_strict_complete": equivalence_complete,
        "claim_status": gate.get("claim_status") or ("robust_allowed" if robust_claim_allowed else "blocked"),
        "precision_winner": global_winner.get("winner") if robust_claim_allowed else None,
        "diagnostic_precision_winner": winner_decision.get("precision_winner"),
        "winner_by_dataset": winner_by_dataset,
        "global_winner": global_winner,
        "diagnostic_winner": {
            "winner": winner_decision.get("precision_winner"),
            "reason": "diagnostic_only_when_robust_claim_blocked" if not robust_claim_allowed else "matches_robust_winner",
        },
        "blockers": unique_blockers,
        "next_actions": _next_actions_from_blockers(unique_blockers),
        "inputs": {
            "final_statistics": str(final_stats_path),
            "gate_status": str(gate_path),
            "release_manifest": str(release_path),
            "final_report": str(report_path),
            "equivalence_strict_summary": str(equivalence_path),
        },
        "release_status": release.get("status") if release else "missing",
        "equivalence_status": equivalence.get("status") if equivalence else "missing",
        "equivalence_run_count": equivalence.get("run_count") if equivalence else 0,
        "equivalence_samples_proven_total": equivalence.get("samples_proven_total") if equivalence else 0,
        "report_claim_status": report.get("claim_status") if report else "missing",
        "outputs": {
            "strict_summary": str(output),
            "strict_summary_md": str(md_output),
            "equivalence_strict_summary": str(equivalence_path),
        },
    }
    write_json(output, payload)
    _write_strict_summary_markdown(md_output, payload)
    stdout = json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return InternalStageResult(
        status="completed" if robust_claim_allowed else "blocked",
        returncode=0 if robust_claim_allowed else 1,
        stdout=stdout,
        stderr="\n".join(payload["blockers"]) + ("\n" if payload["blockers"] else ""),
        blockers=payload["blockers"],
        outputs={"strict_summary": str(output), "strict_summary_md": str(md_output)},
        message="Strict summary generated.",
    )


def execute_internal_stage(stage: str, args: argparse.Namespace) -> InternalStageResult:
    if stage == "equivalence":
        return run_equivalence_stage(args)
    if stage == "summary":
        return run_summary_stage(args)
    return InternalStageResult(status="completed", stdout="Internal stage completed.\n", message="Internal stage completed.")


def parse_stage_list(raw: str | None) -> list[str]:
    if raw in (None, "", "all"):
        return list(STAGE_ORDER)
    requested = [canonical_stage_name(item.strip()) for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(requested) - set(STAGE_ORDER))
    if unknown:
        raise RuntimeError("Unknown pipeline stages: " + ", ".join(unknown))
    requested_set = set(requested)
    return [stage for stage in STAGE_ORDER if stage in requested_set]


def resolve_stage_sequence(args: argparse.Namespace) -> list[str]:
    stages = parse_stage_list(args.stages)
    if args.from_stage:
        args.from_stage = canonical_stage_name(args.from_stage)
        if args.from_stage not in STAGE_ORDER:
            raise RuntimeError(f"Unknown --from-stage: {args.from_stage}")
        stages = [stage for stage in stages if STAGE_ORDER.index(stage) >= STAGE_ORDER.index(args.from_stage)]
    if args.stop_after:
        args.stop_after = canonical_stage_name(args.stop_after)
        if args.stop_after not in STAGE_ORDER:
            raise RuntimeError(f"Unknown --stop-after: {args.stop_after}")
        stages = [stage for stage in stages if STAGE_ORDER.index(stage) <= STAGE_ORDER.index(args.stop_after)]
    if not stages:
        raise RuntimeError("No stages selected after applying --stages/--from-stage/--stop-after.")
    return stages


def resume_status_is_valid(status: str, *, dry_run: bool) -> bool:
    if status == "completed":
        return True
    return bool(dry_run and status == "planned_dry_run")


def should_skip_for_resume(workflow_root: Path, stage: str, *, dry_run: bool) -> bool:
    manifest = read_json(stage_manifest_path(workflow_root, stage))
    return resume_status_is_valid(str(manifest.get("status") or ""), dry_run=dry_run)


def execute_stage(stage: str, args: argparse.Namespace) -> dict[str, Any]:
    workflow_root = args.workflow_root
    workflow_root.mkdir(parents=True, exist_ok=True)
    stdout_log, stderr_log = stage_log_paths(workflow_root, stage)

    if args.resume and should_skip_for_resume(workflow_root, stage, dry_run=bool(args.dry_run)):
        existing = read_json(stage_manifest_path(workflow_root, stage))
        started_at = finished_at = timestamp()
        plan = CommandPlan(
            command=existing.get("command") if isinstance(existing.get("command"), list) else [],
            inputs=existing.get("inputs") if isinstance(existing.get("inputs"), dict) else {},
            outputs=existing.get("outputs") if isinstance(existing.get("outputs"), dict) else {},
            message=f"Skipped by --resume; existing stage status is {existing.get('status')}.",
            heavy=bool(existing.get("heavy")),
            internal=bool(existing.get("internal")),
        )
        ensure_log_files(stdout_log, stderr_log)
        manifest = write_stage_manifest(
            workflow_root=workflow_root,
            stage=stage,
            status="skipped",
            plan=plan,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=0.0,
            returncode=existing.get("returncode") if isinstance(existing.get("returncode"), int) else None,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            message=plan.message,
        )
        append_state(workflow_root, manifest, overall_status="running")
        return manifest

    started_at = timestamp()
    start_time = time.monotonic()
    try:
        plan = build_stage_plan(stage, args)
    except Exception as exc:
        ensure_log_files(stdout_log, stderr_log, stderr=str(exc) + "\n")
        finished_at = timestamp()
        manifest = write_stage_manifest(
            workflow_root=workflow_root,
            stage=stage,
            status="blocked",
            plan=CommandPlan(command=[], message=str(exc)),
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - start_time,
            returncode=None,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            blockers=[str(exc)],
        )
        append_state(workflow_root, manifest, overall_status="failed")
        raise

    if args.dry_run:
        stdout = json.dumps(
            {
                "status": "planned_dry_run",
                "stage": stage,
                "command": plan.command,
                "message": "No subprocess executed because --dry-run is set.",
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
        ensure_log_files(stdout_log, stderr_log, stdout=stdout)
        finished_at = timestamp()
        manifest = write_stage_manifest(
            workflow_root=workflow_root,
            stage=stage,
            status="planned_dry_run",
            plan=plan,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - start_time,
            returncode=None,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            message="Stage planned only; no subprocess executed.",
        )
        append_state(workflow_root, manifest, overall_status="running")
        return manifest

    if plan.internal:
        result = execute_internal_stage(stage, args)
        plan = CommandPlan(
            command=plan.command,
            inputs=plan.inputs,
            outputs={**plan.outputs, **result.outputs},
            env=plan.env,
            message=result.message or plan.message,
            heavy=plan.heavy,
            internal=True,
        )
        ensure_log_files(stdout_log, stderr_log, stdout=result.stdout, stderr=result.stderr)
        finished_at = timestamp()
        manifest = write_stage_manifest(
            workflow_root=workflow_root,
            stage=stage,
            status=result.status,
            plan=plan,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - start_time,
            returncode=result.returncode,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            blockers=result.blockers,
            warnings=result.warnings,
        )
        append_state(workflow_root, manifest, overall_status="running" if result.status == "completed" else "failed")
        if result.status != "completed":
            raise RuntimeError(f"Pipeline stage {stage!r} blocked: " + "; ".join(result.blockers))
        return manifest

    env = os.environ.copy()
    env.update(plan.env)
    completed = subprocess.run(
        plan.command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    ensure_log_files(stdout_log, stderr_log, stdout=completed.stdout, stderr=completed.stderr)
    finished_at = timestamp()
    status = "completed" if completed.returncode == 0 else "failed"
    blockers = [] if completed.returncode == 0 else [f"Command exited with return code {completed.returncode}."]
    manifest = write_stage_manifest(
        workflow_root=workflow_root,
        stage=stage,
        status=status,
        plan=plan,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=time.monotonic() - start_time,
        returncode=completed.returncode,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        blockers=blockers,
    )
    append_state(workflow_root, manifest, overall_status="running" if completed.returncode == 0 else "failed")
    if completed.returncode != 0:
        raise RuntimeError(f"Pipeline stage {stage!r} failed with return code {completed.returncode}.")
    return manifest


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    args.workflow_root = Path(args.workflow_root)
    if args.protocol is not None:
        args.protocol = Path(args.protocol)
    if args.pipeline_config is not None:
        args.pipeline_config = Path(args.pipeline_config)
    if args.dataset_root is not None:
        args.dataset_root = Path(args.dataset_root)
    for attr in (
        "search_manifest",
        "selected_configs",
        "robust_rerun_plan",
        "final_run_root",
        "report_run_root",
        "frozen_split_manifest",
        "graph2mat_result_dir",
        "equivalence_output_dir",
        "summary_output",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(args, attr, Path(value))
    args.deeph_run_root = [Path(value) for value in (args.deeph_run_root or [])]

    stages = resolve_stage_sequence(args)
    args.workflow_root.mkdir(parents=True, exist_ok=True)
    write_json(
        pipeline_state_path(args.workflow_root),
        {
            "schema": SCRIPT_SCHEMA,
            "updated_at": timestamp(),
            "workflow_root": str(args.workflow_root),
            "status": "running",
            "selected_stages": stages,
            "completed_stages": [],
            "stages": {},
            "history": [],
        },
    )
    manifests: list[dict[str, Any]] = []
    try:
        for stage in stages:
            manifests.append(execute_stage(stage, args))
    except Exception:
        raise
    state = read_json(pipeline_state_path(args.workflow_root))
    state["status"] = "completed"
    state["updated_at"] = timestamp()
    state["selected_stages"] = stages
    write_json(pipeline_state_path(args.workflow_root), state)
    return {
        "schema": SCRIPT_SCHEMA,
        "status": "completed",
        "workflow_root": str(args.workflow_root),
        "selected_stages": stages,
        "stage_count": len(stages),
        "stages": manifests,
        "pipeline_state": str(pipeline_state_path(args.workflow_root)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=CLI_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workflow-root", type=Path, required=True)
    parser.add_argument("--stages", default="all", help="Comma-separated stages, or 'all'.")
    parser.add_argument("--from-stage", choices=ALL_STAGE_NAMES, default=None)
    parser.add_argument("--stop-after", choices=ALL_STAGE_NAMES, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)

    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verify-datasets", action="store_true")
    parser.add_argument("--write-dataset-manifests", action="store_true")

    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--search-manifest", type=Path, default=None)
    parser.add_argument("--selected-configs", type=Path, default=None)
    parser.add_argument("--robust-rerun-plan", type=Path, default=None)
    parser.add_argument("--final-run-root", type=Path, default=None)
    parser.add_argument("--report-run-root", type=Path, default=None)
    parser.add_argument("--compute-threshold", type=float, default=None)
    parser.add_argument("--confirm-final-test-open", action="store_true")
    parser.add_argument("--materialize-model", choices=("all", "graph2mat", "deeph"), default="all")
    parser.add_argument("--materialize-jobs", type=int, default=2)
    parser.add_argument("--materialize-limit", type=int, default=None)
    parser.add_argument("--materialize-overwrite", action="store_true")
    parser.add_argument("--materialize-watch", action="store_true")
    parser.add_argument("--materialize-skip-existing", action="store_true")
    parser.add_argument("--materialize-max-cycles", type=int, default=None)
    parser.add_argument("--frozen-split-manifest", type=Path, default=None)
    parser.add_argument("--graph2mat-result-dir", type=Path, default=None)
    parser.add_argument("--deeph-run-root", action="append", default=[])
    parser.add_argument("--deeph-run-glob", action="append", default=[])
    parser.add_argument("--equivalence-output-dir", type=Path, default=None)
    parser.add_argument("--equivalence-jobs", type=int, default=4)
    parser.add_argument("--equivalence-sample-limit", type=int, default=10)
    parser.add_argument("--equivalence-sample-id", action="append", default=[])
    parser.add_argument("--matrix-tolerance", type=float, default=None)
    parser.add_argument("--eigenvalue-tolerance", type=float, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_pipeline(parse_args(argv))
    except Exception as exc:
        print(json.dumps({"schema": SCRIPT_SCHEMA, "status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
