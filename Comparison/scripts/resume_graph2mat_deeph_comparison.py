#!/usr/bin/env python3
"""Resume Graph2Mat-vs-DeepH aggregation once a DeepH evaluation exists."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pipeline_ui  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--deeph-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", default="low_energy_rmse_eV")
    parser.add_argument("--top-percent", type=float, default=100.0)
    parser.add_argument("--top-count", type=int, default=None)
    parser.add_argument("--pipeline-python", default=sys.executable)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def wait_for_deeph_eval(eval_dir: Path, timeout_hours: float, poll_seconds: float) -> Path:
    manifest_path = eval_dir / "deeph_eval_manifest.json"
    start = time.time()
    while not manifest_path.exists():
        elapsed = time.time() - start
        if elapsed > timeout_hours * 3600.0:
            raise TimeoutError(f"Timed out waiting for {manifest_path}")
        print(
            f"[RESUME] waiting {elapsed / 60.0:.1f} min for {manifest_path}",
            flush=True,
        )
        time.sleep(max(1.0, poll_seconds))
    return manifest_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_root = args.output_dir / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)

    if args.wait:
        wait_for_deeph_eval(args.deeph_eval_dir, args.timeout_hours, args.poll_seconds)
    elif not (args.deeph_eval_dir / "deeph_eval_manifest.json").exists():
        raise SystemExit(f"Missing DeepH evaluation manifest: {args.deeph_eval_dir / 'deeph_eval_manifest.json'}")

    manifest = yaml.safe_load(args.experiment_manifest.read_text(encoding="utf-8"))
    runner = pipeline_ui.ExperimentRunner()
    candidates = runner._select_graph2mat_top_candidates(
        manifest,
        args.primary_metric,
        args.top_percent,
        args.top_count,
    )
    selection_rows = []
    for index, candidate in enumerate(candidates, start=1):
        selection_rows.append(
            {
                "rank": index,
                "dataset_label": candidate.get("dataset_label"),
                "result_dir": candidate.get("result_dir"),
                "selection_metric": candidate.get("_deeph_selection_metric"),
                "selection_score": candidate.get("_deeph_selection_score"),
                "split_manifest_hash": candidate.get("split_manifest_hash"),
            }
        )
    pipeline_ui.write_csv_dicts(comparison_root / "selected_graph2mat_candidates.csv", selection_rows)
    (comparison_root / "selected_graph2mat_candidates.json").write_text(
        json.dumps(pipeline_ui.json_safe(selection_rows), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    outputs = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_result_dir = Path(str(candidate["result_dir"]))
        label = pipeline_ui.compact_dataset_label(
            f"g2m_top{index}_{candidate.get('dataset_label') or candidate_result_dir.parent.name}",
            {"rank": index, "result_dir": str(candidate_result_dir)},
            max_length=96,
        )
        candidate_out = comparison_root / label
        command = [
            str(args.pipeline_python),
            str(SCRIPT_DIR / "compare_graph2mat_deeph.py"),
            "--graph2mat-result-dir",
            str(candidate_result_dir),
            "--deeph-eval-dir",
            str(args.deeph_eval_dir),
            "--output-dir",
            str(candidate_out),
        ]
        print(f"[RESUME] compare top {index}/{len(candidates)}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        candidate_out.mkdir(parents=True, exist_ok=True)
        (candidate_out / "compare_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (candidate_out / "compare_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise SystemExit(
                f"compare_graph2mat_deeph failed for {candidate_result_dir} "
                f"with return code {completed.returncode}; see {candidate_out}"
            )
        outputs.append(
            {
                "rank": index,
                "dataset_label": candidate.get("dataset_label"),
                "result_dir": str(candidate_result_dir),
                "comparison_dir": str(candidate_out),
                "report": str(candidate_out / "final_report.md"),
                "aggregate_csv": str(candidate_out / "aggregate_graph2mat_vs_deeph.csv"),
                "selection_metric": candidate.get("_deeph_selection_metric"),
                "selection_score": candidate.get("_deeph_selection_score"),
            }
        )

    pipeline_ui.write_csv_dicts(comparison_root / "comparison_outputs.csv", outputs)
    (comparison_root / "comparison_outputs.json").write_text(
        json.dumps(pipeline_ui.json_safe(outputs), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "resume_manifest.json").write_text(
        json.dumps(
            {
                "experiment_manifest": str(args.experiment_manifest),
                "deeph_eval_dir": str(args.deeph_eval_dir),
                "graph2mat_candidates_compared": len(candidates),
                "comparison_outputs": outputs,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[RESUME] done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
