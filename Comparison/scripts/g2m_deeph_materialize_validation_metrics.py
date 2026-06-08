#!/usr/bin/env python3
"""Materialize split metrics for completed final-seed reruns.

By default this evaluates the validation split only. Locked-test evaluation must
be requested explicitly with ``--split test --protocol-stage final_test`` and a
confirmation flag; that mode still reuses completed training outputs and does
not retrain.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from g2m_deeph_metrics import stage_deeph_metric_inputs, stage_graph2mat_metric_result  # noqa: E402
from g2m_deeph_live_metrics import completed_metric_record  # noqa: E402
from g2m_deeph_runner import (  # noqa: E402
    DEFAULT_HAMILTONIAN_METRICS_SCRIPT,
    DEFAULT_MD_PREDICTION_SCRIPT,
    Graph2MatDeepHBenchmarkRunner,
    _deeph_metric_command_args,
    _extract_validation_metrics,
    _load_json,
    _metric_allowed_returncodes,
    _metric_fail_policy,
    _write_json,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_payload(path: Path | None, *, run_root: Path, split: str, protocol_stage: str) -> dict[str, Any]:
    if path is not None:
        payload = read_json(path)
    else:
        workflow_root = run_root.parents[1]
        candidate = workflow_root / "final" / "run_final_payload.json"
        if not candidate.exists():
            raise RuntimeError(f"Could not infer run-final payload. Pass --payload explicitly; missing {candidate}")
        payload = read_json(candidate)
    payload = dict(payload)
    payload["run_id"] = run_root.name
    payload["output_root"] = str(run_root.parent)
    payload["reuse_run_root"] = True
    payload["search_validation_metrics"] = split == "validation"
    payload["metric_evaluation_split"] = split
    payload["protocol_stage"] = protocol_stage
    return payload


def load_training_sweep(run_root: Path) -> dict[str, Any]:
    path = run_root / "sweep" / "training_sweep_manifest.json"
    if not path.exists():
        raise RuntimeError(f"Missing training sweep manifest: {path}")
    return read_json(path)


def load_working_training_sweep(run_root: Path, output_run_root: Path | None) -> dict[str, Any]:
    if output_run_root is not None:
        output_path = output_run_root / "sweep" / "training_sweep_manifest.json"
        if output_path.exists():
            return read_json(output_path)
    return load_training_sweep(run_root)


def write_training_sweep(run_root: Path, manifest: dict[str, Any]) -> None:
    manifest["validation_metrics_materialized_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(run_root / "sweep" / "training_sweep_manifest.json", manifest)


def metric_record_is_complete(record: dict[str, Any], *, split: str) -> bool:
    if split == "test":
        if isinstance(record.get("final_test_metrics"), dict) and record.get("final_test_metrics"):
            return True
        raw_path = str(record.get("final_test_metrics_path") or "").strip()
        if not raw_path:
            return False
        return Path(raw_path).exists()
    return completed_metric_record(record)


def metric_output_fields(
    *,
    record: dict[str, Any],
    metrics: dict[str, float],
    metrics_path: Path,
    split: str,
    protocol_stage: str,
    primary_metric: str | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "metric_split": split,
        "protocol_stage": protocol_stage,
        "metrics_path": str(metrics_path),
    }
    if split == "test":
        fields.update(
            {
                "final_test_metrics": metrics,
                "final_test_metrics_path": str(metrics_path),
                "final_test_metrics_materialized": True,
            }
        )
        if primary_metric:
            value = metrics.get(primary_metric)
            if value is not None:
                fields["final_test_metric_value"] = value
                fields[primary_metric] = value
    else:
        fields.update(
            {
                "validation_metrics": metrics,
                "validation_metrics_path": str(metrics_path),
                "validation_metrics_materialized": True,
            }
        )
    return fields


def primary_metric_from_payload(payload: dict[str, Any]) -> str | None:
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    final_eval = protocol.get("final_evaluation") if isinstance(protocol.get("final_evaluation"), dict) else {}
    value = str(final_eval.get("primary_metric") or "").strip()
    return value or None


def split_output_name(split: str, default: str, test_name: str) -> str:
    return default if split == "validation" else test_name


def graph2mat_predictions_complete(runner: Graph2MatDeepHBenchmarkRunner, context: Any) -> dict[str, Any] | None:
    try:
        return runner._validate_graph2mat_prediction_outputs(context)
    except RuntimeError:
        return None


def deeph_prediction_present(work_dir: Path) -> bool:
    candidates = [Path(work_dir) / "hamiltonians_pred.h5", Path(work_dir) / "rh_pred.h5"]
    return any(path.exists() and path.stat().st_size > 0 for path in candidates)


def missing_deeph_inference_configs(context: Any) -> list[tuple[Path, Path]]:
    configs = [Path(path) for path in (getattr(context, "inference_configs", []) or [])]
    work_dirs = [Path(path) for path in (getattr(context, "inference_work_dirs", []) or [])]
    return [
        (config, work_dir)
        for config, work_dir in zip(configs, work_dirs, strict=False)
        if not deeph_prediction_present(work_dir)
    ]


def deeph_predictions_complete(context: Any) -> bool:
    work_dirs = list(getattr(context, "inference_work_dirs", []) or [])
    if not work_dirs:
        return False
    return not missing_deeph_inference_configs(context)


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("model") or ""),
        str(record.get("dataset_id") or ""),
        str(record.get("config_id") or ""),
    )


def materialize_graph2mat(
    *,
    runner: Graph2MatDeepHBenchmarkRunner,
    payload: dict[str, Any],
    validation: dict[str, Any],
    record: dict[str, Any],
    split: str,
    protocol_stage: str,
    primary_metric: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    child = runner._child_training_payload(payload, record)
    child["reuse_run_root"] = True
    child["search_validation_metrics"] = split == "validation"
    child["metric_evaluation_split"] = split
    child["protocol_stage"] = protocol_stage
    context = runner._prepare_graph2mat_context(child, {**validation, "dataset_root": record["dataset_root"]})
    checkpoint_manifest = _load_json(context.training_dir / "checkpoint_manifest.json")
    prediction_outputs = graph2mat_predictions_complete(runner, context)
    if prediction_outputs is None:
        predict_run = runner._run_command(
            [runner._graph2mat_python(child), str(DEFAULT_MD_PREDICTION_SCRIPT)],
            cwd=REPO_ROOT,
            env=runner._graph2mat_command_env(context, child),
            label=f"Graph2Mat {split} predict {record['config_id']}",
        )
        prediction_outputs = runner._validate_graph2mat_prediction_outputs(context)
    else:
        predict_run = {
            "label": f"Graph2Mat {split} predict {record['config_id']}",
            "status": "skipped_existing_predictions",
            "returncode": 0,
            "message": "Existing Graph2Mat prediction files were reused.",
        }
    runner._write_graph2mat_manifest(
        context,
        checkpoint_manifest=checkpoint_manifest,
        prediction_outputs=prediction_outputs,
        extra={"prediction_completed": True, "prediction_run": predict_run, "sweep_record": record},
    )
    metrics_root = context.run_root / "metrics" / "graph2mat"
    eval_dir = split_output_name(split, "eval_input", "test_eval_input")
    metric_fail_policy = _metric_fail_policy(child)
    result_dir = metrics_root / eval_dir
    metrics_manifest = result_dir / "metrics" / "manifest.json"
    if metrics_manifest.exists() and not overwrite:
        metrics_run = {
            "label": f"Graph2Mat validation metrics {record['config_id']}",
            "status": "skipped_existing_metrics",
            "returncode": 0,
            "message": "Existing Graph2Mat metric files were reused.",
        }
    else:
        staged = stage_graph2mat_metric_result(
            frozen_split_manifest=_load_json(context.frozen_split_manifest_path),
            prediction_structs_dir=context.prediction_structs_dir,
            output_dir=result_dir,
            dataset_root=context.dataset_root,
            split=split,
        )
        result_dir = staged.result_dir
        metrics_run = runner._run_command(
            [
                runner._graph2mat_python(child),
                str(DEFAULT_HAMILTONIAN_METRICS_SCRIPT),
                str(result_dir),
                "--workers",
                "1",
                "--enable-kpoint-metrics",
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            label=f"Graph2Mat validation metrics {record['config_id']}",
            allowed_returncodes=_metric_allowed_returncodes(metric_fail_policy),
        )
    metrics = _extract_validation_metrics(result_dir / "metrics")
    runner._write_graph2mat_manifest(
        context,
        checkpoint_manifest=checkpoint_manifest,
        prediction_outputs=prediction_outputs,
        extra={
            "prediction_completed": True,
            "prediction_run": predict_run,
            "metrics_run": metrics_run,
            "sweep_record": record,
            **metric_output_fields(
                record=record,
                metrics=metrics,
                metrics_path=result_dir / "metrics" / "manifest.json",
                split=split,
                protocol_stage=protocol_stage,
                primary_metric=primary_metric,
            ),
        },
    )
    return {
        **record,
        "status": "completed",
        "predict_run": predict_run,
        "metrics_run": metrics_run,
        **metric_output_fields(
            record=record,
            metrics=metrics,
            metrics_path=result_dir / "metrics" / "manifest.json",
            split=split,
            protocol_stage=protocol_stage,
            primary_metric=primary_metric,
        ),
    }


def materialize_deeph(
    *,
    runner: Graph2MatDeepHBenchmarkRunner,
    payload: dict[str, Any],
    validation: dict[str, Any],
    record: dict[str, Any],
    split: str,
    protocol_stage: str,
    primary_metric: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    child = runner._child_training_payload(payload, record)
    child["reuse_run_root"] = True
    child["search_validation_metrics"] = split == "validation"
    child["metric_evaluation_split"] = split
    child["protocol_stage"] = protocol_stage
    graph_context = runner._prepare_graph2mat_context(child, {**validation, "dataset_root": record["dataset_root"]})
    deeph_context = runner._prepare_deeph_context(child, graph_context)
    training_outputs = runner._validate_deeph_training_outputs(deeph_context)
    staged_inputs = runner._stage_deeph_inference_inputs(deeph_context)
    inference_runs = []
    missing_inference = missing_deeph_inference_configs(deeph_context)
    skipped_count = len(getattr(deeph_context, "inference_configs", []) or []) - len(missing_inference)
    if not missing_inference:
        inference_runs.append(
            {
                "label": f"DeepH {split} inference {record['config_id']}",
                "status": "skipped_existing_predictions",
                "returncode": 0,
                "message": "Existing DeepH prediction files were reused.",
            }
        )
    else:
        if skipped_count > 0:
            inference_runs.append(
                {
                    "label": f"DeepH {split} inference {record['config_id']} existing predictions",
                    "status": "skipped_existing_predictions",
                    "returncode": 0,
                    "message": f"Reused existing DeepH prediction files for {skipped_count} samples.",
                }
            )
        for inference_config, _work_dir in missing_inference:
            inference_runs.append(
                runner._run_command(
                    [runner._deeph_command(child, "deeph-inference"), "--config", str(inference_config)],
                    cwd=deeph_context.root,
                    env=runner._deeph_command_env(child),
                    label=f"DeepH {split} inference {record['config_id']} {inference_config.stem}",
                )
            )
    prediction_outputs = runner._validate_deeph_prediction_outputs(deeph_context)
    runner._write_deeph_manifest(
        deeph_context,
        inference_runs=inference_runs,
        training_outputs=training_outputs,
        prediction_outputs=prediction_outputs,
        extra={"inference_inputs": staged_inputs, "sweep_record": record},
    )
    metrics_root = graph_context.run_root / "metrics" / "deeph"
    reference_name = split_output_name(split, "reference_input", "test_reference_input")
    inputs_name = split_output_name(split, "deeph_inputs", "test_deeph_inputs")
    eval_name = split_output_name(split, "eval", "test_eval")
    reference_dir = runner._stage_reference_metric_result(
        frozen_split_manifest=_load_json(graph_context.frozen_split_manifest_path),
        output_dir=metrics_root / reference_name,
        dataset_root=graph_context.dataset_root,
        split=split,
    )
    staged_deeph = stage_deeph_metric_inputs(
        raw_mirror=deeph_context.raw_mirror,
        processed_dir=deeph_context.processed_dir,
        inference_dir=deeph_context.inference_dir,
        output_dir=metrics_root / inputs_name,
        split=split,
    )
    metric_fail_policy = _metric_fail_policy(child)
    metrics_path = metrics_root / eval_name / "metrics" / "manifest.json"
    if metrics_path.exists() and not overwrite:
        metrics_run = {
            "label": f"DeepH validation metrics {record['config_id']}",
            "status": "skipped_existing_metrics",
            "returncode": 0,
            "message": "Existing DeepH metric files were reused.",
        }
    else:
        metrics_run = runner._run_command(
            _deeph_metric_command_args(
                python_executable=runner._graph2mat_python(child),
                graph2mat_result_dir=reference_dir,
                processed_dir=staged_deeph.processed_dir,
                predictions_dir=staged_deeph.predictions_dir,
                output_dir=metrics_root / eval_name,
                metric_fail_policy=metric_fail_policy,
                split=split,
            ),
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            label=f"DeepH validation metrics {record['config_id']}",
            allowed_returncodes=_metric_allowed_returncodes(metric_fail_policy),
        )
    metrics = _extract_validation_metrics(metrics_root / eval_name / "metrics")
    runner._write_deeph_manifest(
        deeph_context,
        inference_runs=inference_runs,
        training_outputs=training_outputs,
        prediction_outputs=prediction_outputs,
        extra={
            "inference_inputs": staged_inputs,
            "sweep_record": record,
            "metrics_run": metrics_run,
            **metric_output_fields(
                record=record,
                metrics=metrics,
                metrics_path=metrics_path,
                split=split,
                protocol_stage=protocol_stage,
                primary_metric=primary_metric,
            ),
        },
    )
    return {
        **record,
        "status": "completed",
        "inference_runs": inference_runs,
        "prediction_outputs": prediction_outputs,
        "metrics_run": metrics_run,
        **metric_output_fields(
            record=record,
            metrics=metrics,
            metrics_path=metrics_path,
            split=split,
            protocol_stage=protocol_stage,
            primary_metric=primary_metric,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--payload", type=Path, default=None)
    parser.add_argument("--model", choices=["all", "graph2mat", "deeph"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of records to materialize concurrently.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--protocol-stage", choices=["robust_validation", "final_test"], default="robust_validation")
    parser.add_argument(
        "--confirm-final-test-open",
        action="store_true",
        help="Required with --split test; confirms locked test metrics may now be materialized.",
    )
    parser.add_argument(
        "--output-run-root",
        type=Path,
        default=None,
        help="Optional run-root to receive the updated sweep manifest. Per-run metric artifacts stay under --run-root.",
    )
    parser.add_argument("--watch", action="store_true", help="Poll the run and materialize newly completed records.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In watch mode, ignore records that were already completed when the watcher started.",
    )
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Write per-run metric artifacts only; do not rewrite sweep/training_sweep_manifest.json.",
    )
    parser.add_argument("--max-cycles", type=int, default=0, help="Maximum watch cycles; 0 means unlimited.")
    return parser


def materialize_available_records(
    *,
    args: argparse.Namespace,
    run_root: Path,
    payload: dict[str, Any],
    runner: Graph2MatDeepHBenchmarkRunner,
    skip_keys: set[tuple[str, str, str]],
) -> int:
    target_root = (args.output_run_root or run_root).resolve()
    manifest = load_working_training_sweep(run_root, args.output_run_root)
    records = [row for row in manifest.get("runs") or [] if isinstance(row, dict)]
    if args.model != "all":
        records = [row for row in records if row.get("model") == args.model]
    records = [row for row in records if str(row.get("status") or "") == "completed"]
    records = [row for row in records if record_key(row) not in skip_keys]
    if not args.overwrite:
        records = [row for row in records if not metric_record_is_complete(row, split=args.split)]
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    validation = {"dataset_root": payload.get("dataset_root")}
    updated_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    primary_metric = primary_metric_from_payload(payload)

    def materialize_one(index_record: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
        index, record = index_record
        print(f"[validation-metrics] {index}/{len(records)} {record.get('model')} {record.get('config_id')}", flush=True)
        local_runner = Graph2MatDeepHBenchmarkRunner()
        if record.get("model") == "graph2mat":
            return materialize_graph2mat(
                runner=local_runner,
                payload=payload,
                validation=validation,
                record=record,
                split=args.split,
                protocol_stage=args.protocol_stage,
                primary_metric=primary_metric,
                overwrite=args.overwrite,
            )
        elif record.get("model") == "deeph":
            return materialize_deeph(
                runner=local_runner,
                payload=payload,
                validation=validation,
                record=record,
                split=args.split,
                protocol_stage=args.protocol_stage,
                primary_metric=primary_metric,
                overwrite=args.overwrite,
            )
        else:
            return None

    jobs = max(1, int(args.jobs or 1))
    if jobs > 1 and args.watch:
        raise RuntimeError("--jobs > 1 is not supported with --watch.")
    if jobs > 1 and len(records) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(materialize_one, item) for item in enumerate(records, start=1)]
            for future in concurrent.futures.as_completed(futures):
                updated = future.result()
                if updated is None:
                    continue
                updated_by_key[record_key(updated)] = updated
    else:
        for item in enumerate(records, start=1):
            updated = materialize_one(item)
            if updated is None:
                continue
            updated_by_key[record_key(updated)] = updated
            if args.no_update_manifest:
                continue
            all_rows = []
            for row in manifest.get("runs") or []:
                row_key = record_key(row)
                all_rows.append(updated_by_key.get(row_key, row))
            manifest["runs"] = all_rows
            write_training_sweep(target_root, manifest)

    if jobs > 1 and updated_by_key and not args.no_update_manifest:
        all_rows = []
        for row in manifest.get("runs") or []:
            row_key = record_key(row)
            all_rows.append(updated_by_key.get(row_key, row))
        manifest["runs"] = all_rows
        write_training_sweep(target_root, manifest)
    print(f"[validation-metrics] completed {len(updated_by_key)} records", flush=True)
    return len(updated_by_key)


def main() -> int:
    args = build_parser().parse_args()
    if args.split == "test":
        if args.protocol_stage != "final_test":
            raise RuntimeError("--split test requires --protocol-stage final_test.")
        if not args.confirm_final_test_open:
            raise RuntimeError("--split test requires --confirm-final-test-open.")
    if args.protocol_stage == "final_test" and args.split != "test":
        raise RuntimeError("--protocol-stage final_test requires --split test.")
    run_root = args.run_root.resolve()
    if args.output_run_root is not None:
        args.output_run_root.mkdir(parents=True, exist_ok=True)
    payload = load_payload(args.payload, run_root=run_root, split=args.split, protocol_stage=args.protocol_stage)
    runner = Graph2MatDeepHBenchmarkRunner()
    skip_keys: set[tuple[str, str, str]] = set()
    if args.watch and args.skip_existing:
        manifest = load_training_sweep(run_root)
        skip_keys = {
            record_key(row)
            for row in manifest.get("runs") or []
            if isinstance(row, dict) and str(row.get("status") or "") == "completed"
        }
        print(f"[validation-metrics] watch skip_existing={len(skip_keys)} records", flush=True)
    cycles = 0
    while True:
        materialize_available_records(
            args=args,
            run_root=run_root,
            payload=payload,
            runner=runner,
            skip_keys=skip_keys,
        )
        if not args.watch:
            break
        cycles += 1
        if args.max_cycles and cycles >= args.max_cycles:
            break
        time.sleep(max(1.0, float(args.poll_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
