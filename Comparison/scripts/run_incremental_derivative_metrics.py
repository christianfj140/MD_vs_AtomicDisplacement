#!/usr/bin/env python3
"""Run missing derivative metric postprocess steps for completed workflows."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hamiltonian_derivative_metrics import evaluate_derivative_metrics  # noqa: E402
from g2m_deeph_derivative_gate_check import build_derivative_gate_report, write_json as write_gate_json  # noqa: E402
from g2m_deeph_runner import build_derivative_model_comparison_summary  # noqa: E402
from plot_hamiltonian_derivative_metrics import write_derivative_plot_outputs  # noqa: E402


SIZES = (20, 40, 60, 80, 100, 200, 300, 500, 800, 1100)
MODELS = ("graph2mat", "deeph")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def derivative_workflows_root(run_root: Path) -> Path:
    for candidate in (run_root / "derivative_workflows", run_root / "sweep" / "derivative_workflows"):
        if candidate.exists():
            return candidate
    return run_root / "derivative_workflows"


def workflow_root(run_root: Path, size: int) -> Path:
    return derivative_workflows_root(run_root) / f"graphene_w90_scale_iid{size}"


def manifest_complete(path: Path) -> bool:
    payload = read_json(path)
    if not payload:
        return False
    total = payload.get("samples_total")
    ok = payload.get("samples_ok")
    failed = payload.get("samples_failed", 0)
    try:
        return int(total) > 0 and int(ok) == int(total) and int(failed or 0) == 0
    except (TypeError, ValueError):
        return False


def dependency_paths(wf: Path) -> dict[str, Path]:
    return {
        "stencil": wf / "derivative_stencil_manifest.json",
        "siesta": wf / "siesta_hamiltonians" / "derivative_siesta_reference_manifest.json",
        "graph2mat": wf / "graph2mat_derivative_result" / "predicted_hamiltonians" / "derivative_graph2mat_prediction_manifest.json",
        "deeph": wf / "deeph_derivative_result" / "predicted_hamiltonians" / "derivative_deeph_prediction_manifest.json",
    }


def active_process_lines(wf: Path) -> list[str]:
    try:
        output = subprocess.run(
            ["ps", "-eo", "pid,ppid,etime,%cpu,%mem,stat,cmd", "--sort=-%cpu"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except OSError:
        return []
    needles = {str(wf), wf.name}
    return [line for line in output.splitlines() if any(needle in line for needle in needles)]


def result_dir_for(wf: Path, model: str) -> Path:
    manifest = read_json(dependency_paths(wf)[model])
    raw = str(manifest.get("stencil_root") or "").strip()
    if raw:
        return Path(raw)
    return wf / f"{model}_derivative_result"


def metrics_root(wf: Path, model: str) -> Path:
    return wf / "derivative_metrics" / model


def metrics_exist(root: Path) -> bool:
    return (root / "manifest.json").exists() and (root / "derivative_matrix_metrics.csv").exists()


def status_for_dependencies(paths: dict[str, Path]) -> str:
    if not paths["stencil"].exists():
        return "missing_workflow"
    if not manifest_complete(paths["siesta"]):
        return "pending_siesta"
    if not manifest_complete(paths["graph2mat"]):
        return "pending_graph2mat_prediction"
    if not manifest_complete(paths["deeph"]):
        return "pending_deeph_prediction"
    return "ready_for_metrics"


def run_incremental_derivative_metrics_for_workflow(
    wf: Path,
    *,
    size: int,
    skip_active: bool = False,
    overwrite_missing_only: bool = True,
) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset_size": size, "workflow_root": str(wf), "status": "pending"}
    if not wf.exists():
        row["status"] = "missing_workflow"
        return row
    active = active_process_lines(wf)
    if skip_active and active:
        row.update({"status": "skipped_active", "active_processes": active[:10]})
        write_json(wf / "derivative_incremental_status.json", row)
        return row

    paths = dependency_paths(wf)
    status = status_for_dependencies(paths)
    if status != "ready_for_metrics":
        row["status"] = status
        write_json(wf / "derivative_incremental_status.json", row)
        return row

    model_roots: dict[str, Path] = {}
    commands: list[dict[str, Any]] = []
    failed = False
    for model in MODELS:
        out = metrics_root(wf, model)
        model_roots[model] = out
        if overwrite_missing_only and metrics_exist(out):
            commands.append({"model": model, "status": "skipped_existing", "output_dir": str(out)})
            continue
        try:
            manifest = evaluate_derivative_metrics(
                result_dir_for(wf, model),
                method="central",
                split="test",
                require_central=True,
                overwrite=False,
                diagnostic_only=True,
                support_threshold=1e-12,
                output_dir=out,
                source_model=model,
            )
            commands.append({"model": model, "status": "completed", "output_dir": str(out), "stencils_ok": manifest.get("stencils_ok")})
        except Exception as exc:
            failed = True
            commands.append({"model": model, "status": "failed", "output_dir": str(out), "error": str(exc)})

    ready_roots = [root for root in model_roots.values() if metrics_exist(root)]
    summary_root = wf / "derivative_metrics" / "summary"
    if len(ready_roots) == 2:
        gate = build_derivative_gate_report(derivative_roots=ready_roots, run_root=wf)
        write_gate_json(summary_root / "derivative_gate_report.json", gate)
        build_derivative_model_comparison_summary(
            graph2mat_root=model_roots["graph2mat"],
            deeph_root=model_roots["deeph"],
            output_dir=summary_root / "derivative_model_comparison",
            gate_report=gate,
        )
        write_derivative_plot_outputs(
            derivative_roots=ready_roots,
            graph2mat_root=model_roots["graph2mat"],
            deeph_root=model_roots["deeph"],
            output_dir=summary_root / "derivative_plots",
        )

    row.update(
        {
            "status": "failed" if failed else "metrics_completed" if len(ready_roots) == 2 else "metrics_partial",
            "commands": commands,
            "graph2mat_metrics": str(model_roots["graph2mat"]),
            "deeph_metrics": str(model_roots["deeph"]),
        }
    )
    write_json(wf / "derivative_incremental_status.json", row)
    return row


def refresh_aggregate_plots(run_root: Path) -> None:
    roots: list[Path] = []
    workflows = derivative_workflows_root(run_root)
    for wf in sorted(workflows.glob("graphene_w90_scale_iid*")):
        for model in MODELS:
            root = metrics_root(wf, model)
            if metrics_exist(root):
                roots.append(root)
    if roots:
        write_derivative_plot_outputs(derivative_roots=roots, output_dir=run_root / "summary" / "derivative_plots")


def run_incremental(run_root: Path, *, sizes: list[int], skip_active: bool, overwrite_missing_only: bool) -> dict[str, Any]:
    rows = [
        run_incremental_derivative_metrics_for_workflow(
            workflow_root(run_root, size),
            size=size,
            skip_active=skip_active,
            overwrite_missing_only=overwrite_missing_only,
        )
        for size in sizes
    ]
    refresh_aggregate_plots(run_root)
    summary = {
        "schema": "incremental_derivative_metrics_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(run_root),
        "rows": rows,
        "completed": len([row for row in rows if row.get("status") == "metrics_completed"]),
        "failed": len([row for row in rows if row.get("status") == "failed"]),
    }
    write_json(run_root / "summary" / "incremental_derivative_metrics_summary.json", summary)
    write_csv(run_root / "summary" / "incremental_derivative_metrics_summary.csv", rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    parser.add_argument("--skip-active", action="store_true")
    parser.add_argument("--no-siesta", action="store_true", help="Accepted for safety; SIESTA is never run by this script.")
    parser.add_argument("--no-predictions", action="store_true", help="Accepted for safety; predictions are never run by this script.")
    parser.add_argument("--overwrite-missing-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_incremental(
        args.run_root,
        sizes=list(args.sizes or SIZES),
        skip_active=bool(args.skip_active),
        overwrite_missing_only=bool(args.overwrite_missing_only),
    )
    print(json.dumps({"summary": str(args.run_root / "summary" / "incremental_derivative_metrics_summary.json"), "completed": summary["completed"], "failed": summary["failed"]}, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
