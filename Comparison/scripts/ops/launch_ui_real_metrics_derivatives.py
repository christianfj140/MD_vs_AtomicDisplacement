#!/usr/bin/env python3
"""Run derivative-only workflows for the completed pairs shown by the UI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402
from plot_hamiltonian_derivative_metrics import write_derivative_plot_outputs  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return data


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _graph2mat_info(run_root: Path) -> tuple[Path, Path, Path]:
    manifest_path = run_root / "graph2mat" / "graph2mat_manifest.json"
    manifest = _read(manifest_path)
    checkpoint = Path(str((manifest.get("checkpoint_manifest") or {}).get("checkpoint_path") or ""))
    dataset_root = Path(str((manifest.get("context") or {}).get("dataset_root") or ""))
    if not checkpoint.is_file() or not dataset_root.is_dir():
        raise RuntimeError(f"Incomplete Graph2Mat artifacts: {run_root}")
    return checkpoint, dataset_root, manifest_path


def _deeph_model_dir(run_root: Path) -> Path:
    model_dir = run_root / "deeph" / "train"
    if not (model_dir / "best_state_dict.pkl").is_file():
        raise RuntimeError(f"Missing DeepH best_state_dict.pkl: {run_root}")
    return model_dir


def _mixing_cases(summary: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, float], dict[str, Path]] = {}
    for row in summary.get("records") or []:
        if not isinstance(row, dict) or not row.get("output_root"):
            continue
        model = str(row.get("model") or "")
        if model not in {"graph2mat", "deeph"}:
            continue
        key = (int(row["size"]), str(row["mode"]), float(row["ratio"]))
        grouped.setdefault(key, {})[model] = _path(str(row["output_root"]))
    cases: list[dict[str, Any]] = []
    for (size, mode, ratio), roots in sorted(grouped.items()):
        if set(roots) != {"graph2mat", "deeph"}:
            continue
        checkpoint, dataset_root, _ = _graph2mat_info(roots["graph2mat"])
        cases.append({
            "id": f"mixing_size{size}_{mode}_r{ratio:g}",
            "campaign": "mixing",
            "dataset_root": dataset_root,
            "graph2mat_checkpoint": checkpoint,
            "deeph_model_dir": _deeph_model_dir(roots["deeph"]),
            "dataset_size_metadata": {"dataset_root": str(dataset_root), "dataset_id": dataset_root.name, "mode": mode, "ratio": ratio},
        })
    return cases


def _cross_cases(summary: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in summary.get("permutations") or []:
        status = str(row.get("status") or "")
        runner_status = ((row.get("launch") or {}).get("runner_status") or {})
        run_root_value = runner_status.get("run_root")
        if status not in {"trained", "evaluated"} or not run_root_value:
            continue
        run_root = _path(str(run_root_value))
        checkpoint, dataset_root, _ = _graph2mat_info(run_root)
        case_id = str(row.get("payload_id") or run_root.parent.name)
        cases.append({
            "id": f"cross_{case_id}",
            "campaign": "cross_testing",
            "dataset_root": dataset_root,
            "graph2mat_checkpoint": checkpoint,
            "deeph_model_dir": _deeph_model_dir(run_root),
            "dataset_size_metadata": {"dataset_root": str(dataset_root), "dataset_id": case_id},
        })
    return cases


def _payload(
    case: dict[str, Any],
    settings: dict[str, Any],
    output_root: Path,
    stages_override: dict[str, bool] | None = None,
) -> dict[str, Any]:
    result_dir = output_root / str(case["campaign"]) / str(case["id"])
    derivative = {
        **settings,
        "enabled": True,
        "source_dataset_root": str(case["dataset_root"]),
        "frozen_split": str(Path(case["dataset_root"]) / "frozen_split_manifest.json"),
        "output_root": str(result_dir),
        "graph2mat_checkpoint": str(case["graph2mat_checkpoint"]),
        "deeph_model_dir": str(case["deeph_model_dir"]),
        "basis_files": str(Path(case["dataset_root"]) / "material_basis" / "*.ion.xml"),
        "siesta_command": "/home/christian/bin/siesta",
        "deeph_command": "/home/christian/repositorios/DeepH-pack/.venv/bin/deeph-inference",
    }
    return {
        "run_id": str(case["id"]),
        "dataset_mode": "reuse_validated",
        "dataset_root": str(case["dataset_root"]),
        "workflow_mode": "h_then_derivative_full",
        "metric_fail_policy": "fail_closed",
        "stages": {
            "generate_or_validate_dataset": False,
            "freeze_splits": False,
            "train_graph2mat": False,
            "predict_graph2mat": False,
            "train_deeph": False,
            "predict_deeph": False,
            "hamiltonian_metrics": False,
            "build_derivative_stencils": True,
            "validate_derivative_stencils": True,
            "run_derivative_siesta_reference": True,
            "predict_derivative_graph2mat": True,
            "predict_derivative_deeph": True,
            "derivative_metrics_graph2mat": True,
            "derivative_metrics_deeph": True,
            "derivative_gate_check": True,
            "derivative_plots": True
        } if stages_override is None else stages_override,
        "derivative": derivative,
        "derivative_metrics": {"enabled": True, "method": "central", "require_central": True},
        "dataset_size_metadata": case["dataset_size_metadata"],
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _annotate_metric_manifests(result_dir: Path, metadata: dict[str, Any]) -> None:
    for model in ("graph2mat", "deeph"):
        path = result_dir / "derivative_metrics" / model / "manifest.json"
        if not path.exists():
            continue
        manifest = _read(path)
        manifest["dataset_size_metadata"] = metadata
        _write(path, manifest)


def _refresh_plots(output_root: Path, campaigns: set[str]) -> None:
    roots = [
        root
        for campaign in campaigns
        for model in ("graph2mat", "deeph")
        for root in (output_root / campaign).glob(f"*/derivative_metrics/{model}")
        if (root / "manifest.json").exists()
    ]
    if not roots:
        return
    plots = write_derivative_plot_outputs(
        derivative_roots=roots,
        output_dir=output_root / "derivative_metrics" / "summary" / "derivative_plots",
    )
    for result_dir in {root.parent.parent for root in roots}:
        summary_dir = result_dir / "derivative_metrics" / "summary" / "derivative_plots"
        payload_path = summary_dir / "derivative_plot_payload.json"
        manifest_path = summary_dir / "derivative_plot_manifest.json"
        _write(payload_path, plots["payload"])
        _write(
            manifest_path,
            {
                **plots["manifest"],
                "plot_payload": str(payload_path),
                "outputs": {"plot_payload": str(payload_path)},
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    config = _read(_path(args.payload))
    output_root = _path(str(config["output_root"]))
    cases = _mixing_cases(_read(_path(str(config["mixing_summary"]))))
    cases.extend(_cross_cases(_read(_path(str(config["cross_summary"])))))
    selected_ids = {str(value) for value in config.get("include_case_ids") or []}
    if selected_ids:
        available_ids = {str(case["id"]) for case in cases}
        unknown_ids = selected_ids - available_ids
        if unknown_ids:
            raise RuntimeError(f"Unknown derivative campaign case ids: {sorted(unknown_ids)}")
        cases = [case for case in cases if str(case["id"]) in selected_ids]
    output_campaign = str(config.get("output_campaign") or "").strip()
    if output_campaign:
        for case in cases:
            case["campaign"] = output_campaign
    campaigns = {str(case["campaign"]) for case in cases}
    if args.plots_only:
        _refresh_plots(output_root, campaigns)
        return 0
    deeph_autograd_model_subdir = str(config.get("deeph_autograd_model_subdir") or "").strip()
    if deeph_autograd_model_subdir:
        for case in cases:
            case["deeph_model_dir"] = output_root / str(case["campaign"]) / str(case["id"]) / deeph_autograd_model_subdir / "train"
    settings = dict(config["derivative"])
    state_id = str(config.get("campaign_state_id") or "").strip()
    plan_path = output_root / (f"derivative_campaign_plan_{state_id}.json" if state_id else "derivative_campaign_plan.json")
    status_path = output_root / (f"derivative_campaign_status_{state_id}.json" if state_id else "derivative_campaign_status.json")
    plan = {"schema": "ui_real_metrics_derivative_plan_v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": [{**case, "dataset_root": str(case["dataset_root"]), "graph2mat_checkpoint": str(case["graph2mat_checkpoint"]), "deeph_model_dir": str(case["deeph_model_dir"])} for case in cases]}
    _write(plan_path, plan)
    print(f"[DERIVATIVES] planned={len(cases)} output_root={output_root}", flush=True)
    if args.plan_only:
        return 0
    reference_only_stages = {
        "generate_or_validate_dataset": False,
        "freeze_splits": False,
        "train_graph2mat": False,
        "predict_graph2mat": False,
        "train_deeph": False,
        "predict_deeph": False,
        "hamiltonian_metrics": False,
        "build_derivative_stencils": True,
        "validate_derivative_stencils": True,
        "run_derivative_siesta_reference": True,
        "predict_derivative_graph2mat": False,
        "predict_derivative_deeph": False,
        "derivative_metrics_graph2mat": False,
        "derivative_metrics_deeph": False,
        "derivative_gate_check": False,
        "derivative_plots": False,
    } if args.reference_only else None
    completed: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[DERIVATIVES] {index}/{len(cases)} {case['id']}", flush=True)
        runner = Graph2MatDeepHBenchmarkRunner()
        runner.start(_payload(case, settings, output_root, stages_override=reference_only_stages))
        offset = 0
        while runner.status().get("running"):
            logs = runner.logs(since=offset, limit=None)
            for line in logs.get("lines") or []:
                print(line, end="", flush=True)
            offset = int(logs.get("offset") or offset)
            time.sleep(10)
        status = runner.status()
        _annotate_metric_manifests(
            output_root / str(case["campaign"]) / str(case["id"]),
            dict(case["dataset_size_metadata"]),
        )
        completed.append({"id": case["id"], "campaign": case["campaign"], "status": status, "result": runner.results()})
        _write(status_path, {"plan": str(plan_path), "completed": completed})
    _refresh_plots(output_root, campaigns)
    return 0 if all(item["status"].get("returncode") == 0 for item in completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
