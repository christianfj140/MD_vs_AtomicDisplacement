#!/usr/bin/env python3
"""Tiny Graph2Mat vs DeepH smoke entrypoint.

The default mode is a dry-run that exercises the dedicated benchmark runner
without launching SIESTA, Graph2Mat training, or DeepH. The real tiny smoke is
available only behind RUN_G2M_DEEPH_REAL_SMOKE=1 and explicit --tiny-real.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from g2m_deeph_final_workflow import parse_args as workflow_parse_args  # noqa: E402
from g2m_deeph_final_workflow import run_stage as workflow_run_stage  # noqa: E402
from g2m_deeph_gate_check import build_gate_status  # noqa: E402
from g2m_deeph_release_manifest import build_release_manifest  # noqa: E402
from g2m_deeph_runner import (  # noqa: E402
    DATASET_SWEEP_RUN_MODE,
    DEEPH_CLI_NAMES,
    METRIC_FAIL_POLICY_FAIL_CLOSED,
    Graph2MatDeepHBenchmarkRunner,
)
from joint_artifact_contract import validate_dataset  # noqa: E402


REAL_SMOKE_ENV = "RUN_G2M_DEEPH_REAL_SMOKE"
DEFAULT_SAMPLE_LIMIT = 6
REQUIRED_SNAPSHOT_ARTIFACTS = (
    "RUN.fdf",
    "SystemLabel.TSHS",
    "SystemLabel.TSDE",
    "SystemLabel.HSX",
    "SystemLabel.STRUCT_OUT",
    "SystemLabel.XV",
    "SystemLabel.ORB_INDX",
    "metadata.json",
)
PAPER_WORKFLOW_SCIENTIFIC_STATUS = "not_a_scientific_run"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def required_snapshot_artifacts() -> list[str]:
    return list(REQUIRED_SNAPSHOT_ARTIFACTS)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_root() -> Path:
    return REPO_ROOT / "Comparison" / "results" / f"g2m_deeph_smoke_{_timestamp()}"


def _command_exists(command: str | None) -> tuple[bool, str | None]:
    if not command:
        return False, None
    try:
        executable = shlex.split(command)[0]
    except ValueError:
        executable = command
    if os.sep in executable:
        path = Path(executable).expanduser()
        return path.exists() and os.access(path, os.X_OK), str(path)
    found = shutil.which(executable)
    return found is not None, found


def _dependency_report(payload: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    env_enabled = os.environ.get(REAL_SMOKE_ENV) == "1"
    checks[REAL_SMOKE_ENV] = {
        "available": env_enabled,
        "required_value": "1",
        "actual": os.environ.get(REAL_SMOKE_ENV),
    }
    if not env_enabled:
        reasons.append(f"{REAL_SMOKE_ENV}=1 is required for --tiny-real.")

    siesta_command = os.environ.get("SIESTA_COMMAND") or os.environ.get("SIESTA") or "siesta"
    siesta_ok, siesta_path = _command_exists(siesta_command)
    checks["siesta"] = {
        "available": siesta_ok,
        "command": siesta_command,
        "path": siesta_path,
    }
    if not siesta_ok:
        reasons.append("SIESTA executable was not found via SIESTA_COMMAND, SIESTA, or PATH.")

    graph2mat_ok, graph2mat_path = _command_exists("graph2mat")
    checks["graph2mat"] = {
        "available": graph2mat_ok,
        "command": "graph2mat",
        "path": graph2mat_path,
    }
    if not graph2mat_ok:
        reasons.append("graph2mat CLI was not found in PATH.")

    runner = Graph2MatDeepHBenchmarkRunner()
    discovery = runner._deeph_discovery(payload)  # Discovery helper is intentionally reused here.
    checks["deeph"] = discovery
    commands = discovery.get("commands") or {}
    for command_name in DEEPH_CLI_NAMES:
        command = commands.get(command_name) or {}
        if not command.get("path"):
            reasons.append(str(command.get("error") or f"{command_name} was not found."))

    return {
        "status": "available" if not reasons else "missing",
        "reasons": reasons,
        "checks": checks,
    }


def _dataset_generation_payload(
    *,
    output_root: Path,
    sample_limit: int,
    run_id: str,
    dry_run: bool,
    deeph_repo_path: str | None = None,
) -> dict[str, Any]:
    snapshot_count = max(int(sample_limit), DEFAULT_SAMPLE_LIMIT)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "dataset_root": str(output_root / "datasets" / "g2m_deeph_smoke_joint"),
        "output_root": str(output_root / "runner_results"),
        "dataset_mode": "generate_new",
        "run_mode": DATASET_SWEEP_RUN_MODE,
        "snapshot_count": snapshot_count,
        "dry_run": bool(dry_run),
        "split_mode": "block",
        "metric_fail_policy": METRIC_FAIL_POLICY_FAIL_CLOSED,
    }
    if deeph_repo_path:
        payload["deeph"] = {"repo_path": deeph_repo_path}
    return payload


def _benchmark_payload(
    *,
    output_root: Path,
    dataset_root: Path,
    epochs: int,
    run_id: str,
    deeph_repo_path: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root / "runner_results"),
        "dataset_mode": "reuse_validated",
        "dry_run": False,
        "metric_fail_policy": METRIC_FAIL_POLICY_FAIL_CLOSED,
        "graph2mat_overrides": {"max_epochs": int(epochs)},
        "deeph": {"epochs": int(epochs), "batch_size": 1},
    }
    if deeph_repo_path:
        payload["deeph"]["repo_path"] = deeph_repo_path
    return payload


def _run_runner(payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    runner = Graph2MatDeepHBenchmarkRunner()
    started_at = time.time()
    start_status: dict[str, Any] | None = None
    error: str | None = None
    try:
        start_status = runner.start(payload)
        thread = getattr(runner, "_thread", None)
        if thread is not None:
            thread.join(timeout=timeout_seconds)
            if thread.is_alive():
                runner.stop()
                error = f"Runner did not finish within {timeout_seconds:.1f} seconds."
    except Exception as exc:
        error = str(exc)

    status = runner.status()
    results = runner.results()
    logs_payload = runner.logs(since=0, limit=20000)
    lines = logs_payload.get("lines") or logs_payload.get("entries") or []
    return {
        "payload": payload,
        "start_status": start_status,
        "status": status,
        "results": results.get("results") if isinstance(results, dict) else None,
        "available": bool(results.get("available")) if isinstance(results, dict) else False,
        "logs": lines,
        "error": error or status.get("error"),
        "returncode": status.get("returncode"),
        "elapsed_seconds": time.time() - started_at,
    }


def _planned_artifact_validation(*, mode: str, status: str, real_artifacts_checked: bool) -> dict[str, Any]:
    return {
        "schema": "graph2mat_deeph_smoke_artifact_validation_v1",
        "mode": mode,
        "status": status,
        "real_artifacts_checked": real_artifacts_checked,
        "required_snapshot_artifacts": required_snapshot_artifacts(),
        "checks": [
            {
                "artifact": artifact,
                "required": True,
                "checked": real_artifacts_checked,
                "reason": (
                    "real smoke execution validates generated snapshot files"
                    if real_artifacts_checked
                    else "dry-run/skip mode records the contract without claiming real artifacts exist"
                ),
            }
            for artifact in REQUIRED_SNAPSHOT_ARTIFACTS
        ],
    }


def _write_smoke_outputs(
    *,
    output_root: Path,
    manifest: dict[str, Any],
    artifact_validation: dict[str, Any],
    benchmark_manifest: dict[str, Any],
    recommendation: dict[str, Any],
    logs: list[str],
) -> None:
    write_json(output_root / "smoke_manifest.json", manifest)
    write_json(output_root / "artifact_validation.json", artifact_validation)
    write_json(output_root / "benchmark_manifest.json", benchmark_manifest)
    write_json(output_root / "recommendation.json", recommendation)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "smoke.log").write_text("".join(logs), encoding="utf-8")


def _write_workflow_snapshot(path: Path, *, label: str = "graphene") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text(
        "\n".join(
            [
                f"SystemLabel {label}",
                "SaveHS true",
                "Save.HS T",
                "TS.HS.Save T",
                "TS.DE.Save T",
                "XML.Write T",
                "Write.OrbitalIndex T",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (path / "RUN.out").write_text("Synthetic smoke output\n", encoding="utf-8")
    write_json(path / "metadata.json", {"system_label": label, "smoke_fixture": True})
    for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
        (path / f"{label}{suffix}").write_text(f"synthetic {suffix}\n", encoding="utf-8")


def _write_workflow_split_csv(path: Path, *, sample_id: str, sample_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split = path.stem.replace("_manifest", "")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "sample_dir"])
        writer.writeheader()
        writer.writerow({"sample_id": sample_id, "split": split, "sample_dir": str(sample_dir)})


def _create_paper_workflow_dataset(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "RUN.out").write_text("Synthetic dataset-level SIESTA output\n", encoding="utf-8")
    write_json(
        dataset_root / "material_provenance.json",
        {
            "label": "graphene_smoke",
            "fdf_sha256": "synthetic-fdf-hash",
            "basis_file_sha256": {"C.ion.xml": "synthetic-basis-hash"},
            "pseudopotential_sha256": {"C": "synthetic-pseudo-hash"},
            "siesta_version": "SIESTA 5.4.2-synthetic-smoke",
            "siesta_command_line": "siesta < RUN.fdf",
            "run_out_path": str(dataset_root / "RUN.out"),
            "siesta_returncode": 0,
            "environment": {"python_version": sys.version.split()[0], "platform": sys.platform},
        },
    )
    sample_dirs: list[Path] = []
    for index, split in enumerate(("train", "validation", "test")):
        sample_dir = dataset_root / "samples" / f"s{index}"
        _write_workflow_snapshot(sample_dir)
        sample_dirs.append(sample_dir)
        _write_workflow_split_csv(dataset_root / "splits" / f"{split}_manifest.csv", sample_id=f"s{index}", sample_dir=sample_dir)
    artifact_validation = validate_dataset(dataset_root, snapshot_dirs=sample_dirs).to_dict()
    write_json(dataset_root / "artifact_validation.json", artifact_validation)
    write_benchmark_manifests(
        dataset_root=dataset_root,
        split_root=dataset_root / "splits",
        strict_paper_ready_provenance=True,
    )


def _paper_workflow_protocol(dataset_root: Path) -> dict[str, Any]:
    return {
        "protocol_id": "paper_workflow_smoke_protocol",
        "version": "1.0",
        "datasets": [
            {
                "dataset_id": "joint_smoke",
                "dataset_root": str(dataset_root),
                "benchmark_dataset_manifest": str(dataset_root / "benchmark_dataset_manifest.json"),
                "frozen_split_manifest": str(dataset_root / "frozen_split_manifest.json"),
                "split_root": str(dataset_root / "splits"),
            }
        ],
        "reference_artifacts": {
            "required": required_snapshot_artifacts(),
            "forbidden": ["ML_prediction.HSX"],
            "forbid_as_reference": "ML_prediction.HSX",
        },
        "models": {
            "graph2mat": {
                "enabled": True,
                "search_space": {
                    "optim_lr": {"choices": [0.001]},
                    "batch_size": {"choices": [64]},
                    "max_epochs": {"value": 2},
                    "hidden_irreps": {"choices": ["16x0e + 16x1o + 16x2e"]},
                    "num_interactions": {"value": 2},
                    "correlation": {"value": 2},
                    "max_ell": {"value": 2},
                },
            },
            "deeph": {
                "enabled": True,
                "search_space": {
                    "learning_rate": {"choices": [0.0001]},
                    "batch_size": {"choices": [2]},
                    "epochs": {"value": 2},
                    "atom_fea_len": {"value": 64},
                    "edge_fea_len": {"value": 128},
                    "num_l": {"value": 4},
                    "if_lcmp": {"value": True},
                },
            },
        },
        "selection": {"split": "validation", "metric": "low_energy_rmse_eV", "mode": "min", "source": "validation_only"},
        "early_stopping": {"metric": "low_energy_rmse_eV", "mode": "min", "patience": 1, "min_delta": 0.0, "max_epochs": 2},
        "search_policy": {"strategy": "random", "n_trials_per_model": 1, "random_seed": 1},
        "budget_policy": {"mode": "equal_n_trials", "n_trials_per_model": 1},
        "final_seeds": [0, 1, 2],
        "top_k_selection": {"k_per_model": 1, "split": "validation", "metric": "low_energy_rmse_eV", "uses_test_metrics": False},
        "final_evaluation": {"primary_metric": "low_energy_rmse_eV", "mode": "min", "secondary_metrics": ["h_mae_eV"]},
        "final_test_policy": {
            "policy": "locked_until_final",
            "test_split": "test",
            "locked_during_search": True,
            "evaluate_once_after_selection": True,
        },
        "required_telemetry": [
            "wall_clock_seconds",
            "gpu_hours",
            "peak_gpu_memory_mb",
            "samples_per_second",
            "matrix_blocks_per_second",
            "best_validation_epoch",
        ],
        "deeph_comparability": {
            "adapter_equivalence_policy": "fail_closed_unless_proven",
            "robust_winner_requires_proven_equivalence": True,
            "diagnostic_if_unproven": True,
        },
    }


def _workflow_stage(workflow_root: Path, *args: str) -> dict[str, Any]:
    return workflow_run_stage(workflow_parse_args(["--workflow-root", str(workflow_root), *args]))


def _synthetic_search_record(model: str, *, value: float) -> dict[str, Any]:
    return {
        "status": "completed",
        "model": model,
        "dataset_id": "joint_smoke",
        "dataset_root": "",
        "config_id": f"{model}_smoke_config",
        "config_hash": f"{model}-hash",
        "run_root": f"synthetic/{model}",
        "protocol_stage": "search",
        "metric_split": "validation",
        "low_energy_rmse_eV": value,
        "validation_metrics": {"low_energy_rmse_eV": value},
        "common": {"seed": 0, "epochs": 2},
        "overrides": {"seed_everything": 0} if model == "graph2mat" else {"seed": 0},
    }


def _synthetic_final_rows(robust_plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for planned in robust_plan.get("planned_runs") or []:
        if not isinstance(planned, dict):
            continue
        model = str(planned.get("model") or "")
        seed = planned.get("seed", (planned.get("common") or {}).get("seed"))
        value = 0.20 if model == "graph2mat" else 0.19
        row = {
            **planned,
            "status": "completed",
            "protocol_stage": "final_test",
            "metric_split": "test",
            "low_energy_rmse_eV_mean": value,
            "telemetry": {
                "gpu_hours_total": 0.01 if model == "graph2mat" else 0.02,
                "peak_gpu_memory_mb": 128.0,
                "samples_per_second": 10.0,
                "matrix_blocks_per_second": 100.0,
            },
            "per_system_metrics": [{"sample_id": "s2", "low_energy_rmse_eV_mean": value}],
            "seed": seed,
        }
        if model == "deeph":
            row.update(
                {
                    "adapter_equivalence_status": "diagnostic_local_frame_only",
                    "equivalence_status": "unproven",
                    "comparability_status": "diagnostic_only",
                    "diagnostic_only": True,
                }
            )
        rows.append(row)
    return rows


def run_paper_workflow_dry_run(*, output_root: Path | None = None, run_id: str | None = None) -> dict[str, Any]:
    output_root = (output_root or _default_output_root()).resolve()
    workflow_root = output_root / "paper_workflow"
    dataset_root = output_root / "synthetic_dataset"
    final_run_root = workflow_root / "runs" / "synthetic_final"
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"paper_workflow_smoke_{_timestamp()}"
    logs: list[str] = [
        "[PAPER-SMOKE] control-plane dry run only; no scientific benchmark was run.\n",
        f"[PAPER-SMOKE] output_root={output_root}\n",
    ]

    _create_paper_workflow_dataset(dataset_root)
    protocol_path = output_root / "paper_workflow_protocol.json"
    write_json(protocol_path, _paper_workflow_protocol(dataset_root))
    stage_results: list[dict[str, Any]] = []
    for args in (
        ("--stage", "validate-protocol", "--protocol", str(protocol_path), "--verify-datasets"),
        ("--stage", "generate-search-plan"),
        ("--stage", "run-search", "--dry-run", "--dataset-id", "joint_smoke", "--run-id", f"{run_id}_search"),
    ):
        stage_results.append(_workflow_stage(workflow_root, *args))

    write_json(
        workflow_root / "search" / "training_sweep_manifest.json",
        {"runs": [_synthetic_search_record("graph2mat", value=0.20), _synthetic_search_record("deeph", value=0.19)]},
    )
    stage_results.append(_workflow_stage(workflow_root, "--stage", "select-top-k"))
    stage_results.append(_workflow_stage(workflow_root, "--stage", "generate-final-seeds"))
    stage_results.append(
        _workflow_stage(
            workflow_root,
            "--stage",
            "run-final",
            "--dry-run",
            "--dataset-id",
            "joint_smoke",
            "--run-id",
            f"{run_id}_final",
        )
    )

    robust_plan = json.loads((workflow_root / "selection" / "robust_rerun_plan.json").read_text(encoding="utf-8"))
    write_json(final_run_root / "sweep" / "training_sweep_manifest.json", {"runs": _synthetic_final_rows(robust_plan)})
    stage_results.append(_workflow_stage(workflow_root, "--stage", "evaluate-final-test", "--final-run-root", str(final_run_root)))
    stage_results.append(_workflow_stage(workflow_root, "--stage", "generate-report", "--final-run-root", str(final_run_root)))

    gate_status = build_gate_status(protocol_path=protocol_path, workflow_root=workflow_root, run_root=final_run_root)
    write_json(workflow_root / "gate_status.json", gate_status)
    release_manifest = build_release_manifest(
        dataset_root=dataset_root,
        run_root=final_run_root,
        workflow_root=workflow_root,
        strict=False,
    )
    write_json(workflow_root / "artifact_release_manifest.json", release_manifest)

    summary = {
        "schema": "graph2mat_deeph_paper_workflow_smoke_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "paper_workflow_dry_run",
        "status": "passed_control_plane",
        "ok": True,
        "scientific_status": PAPER_WORKFLOW_SCIENTIFIC_STATUS,
        "robust_claim_allowed": bool(gate_status.get("robust_claim_allowed")),
        "diagnostic_only": True,
        "reason": "Synthetic dry-run smoke checks workflow wiring only; it is not benchmark evidence.",
        "run_id": run_id,
        "protocol": str(protocol_path),
        "workflow_root": str(workflow_root),
        "dataset_root": str(dataset_root),
        "final_run_root": str(final_run_root),
        "stage_results": stage_results,
        "gate_status": {
            "path": str(workflow_root / "gate_status.json"),
            "claim_status": gate_status.get("claim_status"),
            "robust_claim_allowed": gate_status.get("robust_claim_allowed"),
            "blockers": gate_status.get("blockers") or [],
        },
        "release_manifest": {
            "path": str(workflow_root / "artifact_release_manifest.json"),
            "status": release_manifest.get("status"),
            "missing_required": release_manifest.get("missing_required") or [],
        },
        "outputs": {
            "smoke_summary": str(output_root / "smoke_summary.json"),
            "protocol": str(protocol_path),
            "dataset_verification": str(workflow_root / "dataset_verification.json"),
            "search_plan": str(workflow_root / "search" / "search_plan.json"),
            "selected_configs": str(workflow_root / "selection" / "selected_configs.json"),
            "robust_rerun_plan": str(workflow_root / "selection" / "robust_rerun_plan.json"),
            "run_search_manifest": str(workflow_root / "search" / "run_search_manifest.json"),
            "run_final_manifest": str(workflow_root / "final" / "run_final_manifest.json"),
            "final_statistics": str(workflow_root / "final_test" / "final_statistics.json"),
            "report_summary": str(workflow_root / "report" / "report_summary.json"),
            "final_report": str(workflow_root / "report" / "final_report.json"),
            "gate_status": str(workflow_root / "gate_status.json"),
            "release_manifest": str(workflow_root / "artifact_release_manifest.json"),
            "log": str(output_root / "logs" / "paper_workflow_smoke.log"),
        },
    }
    write_json(output_root / "smoke_summary.json", summary)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "paper_workflow_smoke.log").write_text("".join(logs), encoding="utf-8")
    return summary


def _first_generated_dataset_root(runner_result: dict[str, Any]) -> Path | None:
    results = runner_result.get("results") or {}
    sweep = results.get("dataset_sweep") if isinstance(results, dict) else None
    rows = sweep.get("rows") if isinstance(sweep, dict) else None
    if not rows:
        return None
    dataset_root = rows[0].get("dataset_root")
    return Path(str(dataset_root)) if dataset_root else None


def run_smoke(
    *,
    output_root: Path | None = None,
    dry_run: bool = True,
    tiny_real: bool = False,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    epochs: int = 1,
    skip_training: bool = False,
    run_id: str | None = None,
    deeph_repo_path: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    output_root = (output_root or _default_output_root()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mode = "tiny_real" if tiny_real else "dry_run"
    run_id = run_id or f"g2m_deeph_smoke_{mode}_{_timestamp()}"
    logs: list[str] = [
        f"[SMOKE] mode={mode}\n",
        f"[SMOKE] output_root={output_root}\n",
        f"[SMOKE] required_artifacts={', '.join(REQUIRED_SNAPSHOT_ARTIFACTS)}\n",
    ]

    generation_payload = _dataset_generation_payload(
        output_root=output_root,
        sample_limit=sample_limit,
        run_id=f"{run_id}_dataset",
        dry_run=dry_run or not tiny_real,
        deeph_repo_path=deeph_repo_path,
    )

    if tiny_real:
        dependency_report = _dependency_report(generation_payload)
        if dependency_report["status"] != "available":
            logs.append("[SMOKE] tiny-real skipped: missing dependencies or env gate.\n")
            for reason in dependency_report["reasons"]:
                logs.append(f"[SMOKE][SKIP] {reason}\n")
            manifest = {
                "schema": "graph2mat_deeph_smoke_manifest_v1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": mode,
                "status": "skipped",
                "ok": None,
                "skip": True,
                "skip_reasons": dependency_report["reasons"],
                "dependency_report": dependency_report,
                "required_snapshot_artifacts": required_snapshot_artifacts(),
                "outputs": {
                    "smoke_manifest": str(output_root / "smoke_manifest.json"),
                    "artifact_validation": str(output_root / "artifact_validation.json"),
                    "benchmark_manifest": str(output_root / "benchmark_manifest.json"),
                    "recommendation": str(output_root / "recommendation.json"),
                    "log": str(output_root / "logs" / "smoke.log"),
                },
            }
            artifact_validation = _planned_artifact_validation(
                mode=mode,
                status="skipped",
                real_artifacts_checked=False,
            )
            benchmark_manifest = {
                "schema": "graph2mat_deeph_smoke_benchmark_manifest_v1",
                "status": "skipped",
                "dependency_report": dependency_report,
                "required_snapshot_artifacts": required_snapshot_artifacts(),
            }
            recommendation = {
                "schema": "graph2mat_deeph_smoke_recommendation_v1",
                "status": "skipped",
                "winner": None,
                "robust_winner_allowed": False,
                "reason": "Tiny real smoke was not run because dependencies/env gate were missing.",
            }
            _write_smoke_outputs(
                output_root=output_root,
                manifest=manifest,
                artifact_validation=artifact_validation,
                benchmark_manifest=benchmark_manifest,
                recommendation=recommendation,
                logs=logs,
            )
            return manifest

    generation_result = _run_runner(generation_payload, timeout_seconds=timeout_seconds)
    logs.extend(generation_result.get("logs") or [])
    generation_ok = generation_result.get("returncode") == 0 and not generation_result.get("error")

    benchmark_result: dict[str, Any] | None = None
    generated_dataset_root = _first_generated_dataset_root(generation_result)
    if tiny_real and generation_ok and not skip_training and generated_dataset_root is not None:
        benchmark_payload = _benchmark_payload(
            output_root=output_root,
            dataset_root=generated_dataset_root,
            epochs=epochs,
            run_id=f"{run_id}_benchmark",
            deeph_repo_path=deeph_repo_path,
        )
        benchmark_result = _run_runner(benchmark_payload, timeout_seconds=timeout_seconds)
        logs.extend(benchmark_result.get("logs") or [])

    if tiny_real:
        final_returncode = (
            benchmark_result.get("returncode")
            if benchmark_result is not None
            else generation_result.get("returncode")
        )
        status = "passed" if final_returncode == 0 else "failed"
    else:
        status = "dry_run_passed" if generation_ok else "dry_run_failed"

    real_artifacts_checked = bool(tiny_real and generation_ok)
    artifact_validation = _planned_artifact_validation(
        mode=mode,
        status=status,
        real_artifacts_checked=real_artifacts_checked,
    )
    benchmark_manifest = {
        "schema": "graph2mat_deeph_smoke_benchmark_manifest_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "status": status,
        "dataset_generation": generation_result,
        "benchmark_run": benchmark_result,
        "required_snapshot_artifacts": required_snapshot_artifacts(),
        "metric_fail_policy": METRIC_FAIL_POLICY_FAIL_CLOSED,
        "skip_training": bool(skip_training),
    }
    recommendation = {
        "schema": "graph2mat_deeph_smoke_recommendation_v1",
        "status": "diagnostic_only" if tiny_real else "dry_run_only",
        "winner": None,
        "robust_winner_allowed": False,
        "reason": (
            "Tiny smoke is an integration check, not a production benchmark."
            if tiny_real
            else "Dry-run smoke only planned dataset generation and did not execute SIESTA/training."
        ),
    }
    manifest = {
        "schema": "graph2mat_deeph_smoke_manifest_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "status": status,
        "ok": status in {"dry_run_passed", "passed"},
        "skip": False,
        "run_id": run_id,
        "sample_limit": int(sample_limit),
        "effective_snapshot_count": max(int(sample_limit), DEFAULT_SAMPLE_LIMIT),
        "epochs": int(epochs),
        "skip_training": bool(skip_training),
        "required_snapshot_artifacts": required_snapshot_artifacts(),
        "outputs": {
            "smoke_manifest": str(output_root / "smoke_manifest.json"),
            "artifact_validation": str(output_root / "artifact_validation.json"),
            "benchmark_manifest": str(output_root / "benchmark_manifest.json"),
            "recommendation": str(output_root / "recommendation.json"),
            "log": str(output_root / "logs" / "smoke.log"),
        },
        "dataset_generation_returncode": generation_result.get("returncode"),
        "benchmark_returncode": benchmark_result.get("returncode") if benchmark_result else None,
        "generated_dataset_root": str(generated_dataset_root) if generated_dataset_root else None,
    }
    _write_smoke_outputs(
        output_root=output_root,
        manifest=manifest,
        artifact_validation=artifact_validation,
        benchmark_manifest=benchmark_manifest,
        recommendation=recommendation,
        logs=logs,
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny Graph2Mat vs DeepH smoke check. Default is dry-run only."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan the smoke without external executables.")
    mode.add_argument("--tiny-real", action="store_true", help=f"Run real tiny smoke only when {REAL_SMOKE_ENV}=1.")
    mode.add_argument(
        "--paper-workflow-dry-run",
        action="store_true",
        help="Exercise the final benchmark control plane using synthetic fixtures only.",
    )
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=None, help="Directory for smoke outputs.")
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT, help="Tiny snapshot count target.")
    parser.add_argument("--epochs", type=int, default=1, help="Tiny training epoch count for --tiny-real.")
    parser.add_argument("--skip-training", action="store_true", help="For --tiny-real, stop after dataset generation.")
    parser.add_argument("--run-id", default=None, help="Stable smoke run id.")
    parser.add_argument("--deeph-repo-path", default=None, help="Optional DeepH-pack repository path.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Per-runner timeout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.paper_workflow_dry_run:
        manifest = run_paper_workflow_dry_run(output_root=args.output_root, run_id=args.run_id)
        print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
        return 0 if manifest.get("ok") else 1
    tiny_real = bool(args.tiny_real)
    dry_run = not tiny_real
    manifest = run_smoke(
        output_root=args.output_root,
        dry_run=dry_run,
        tiny_real=tiny_real,
        sample_limit=args.sample_limit,
        epochs=args.epochs,
        skip_training=args.skip_training,
        run_id=args.run_id,
        deeph_repo_path=args.deeph_repo_path,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    if manifest.get("status") == "skipped":
        return 0
    return 0 if manifest.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
