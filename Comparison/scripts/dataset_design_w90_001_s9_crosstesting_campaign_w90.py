#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9 -- retrain the 80 distinct (family,dim,domain,density)
w90 pilot combos with cross-testing hyperparameters.

The 80 combos are every sampler-family/dim/R_train_max/density combination
among S9-S4 pilot's best-by-family designs restricted to the three families
that generalize to a 6x6 "move every atom" design
(sobol_sparse/random_cartesian/latin_hypercube -- axial_radial/angular_shell
degenerate to a rigid shift at k=n_atoms, local_pair_modes is k=2-only), so
the same 80 recipes can be run on both w90 (this script) and 6x6
(dataset_design_w90_001_s9_crosstesting_campaign_6x6.py) for a direct,
paired comparison. List + real w90 pilot H-MAE per combo:
Comparison/results/dataset_design_w90_001_s9_chain_logs/crosstesting_campaign_80combos.json

Every design here is an existing S9-S4 trainable pilot candidate, real-SIESTA-
backed already (dataset_design_w90_001_s9_cross_testing_hparam_probe.py's
--source pilot path) -- zero new SIESTA calculations, only new Graph2Mat
training with the cross-testing hyperparameters. GPU-VRAM per job measured at
~1.6-2.7GB (small w90 model/dataset), so this runs with generous parallelism.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COMBOS_PATH = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_chain_logs/crosstesting_campaign_80combos.json"
PROBE_SCRIPT = SCRIPT_DIR / "dataset_design_w90_001_s9_cross_testing_hparam_probe.py"
LOG_DIR = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_chain_logs/w90_campaign_logs"
SUMMARY_PATH = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_chain_logs/w90_campaign_summary.json"


def run_one(combo: dict, python_exe: str) -> dict:
    did = combo["w90_design_id"]
    n_train = combo["density"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{did}.log"
    start = time.perf_counter()
    with log_path.open("w") as log_file:
        completed = subprocess.run(
            [python_exe, str(PROBE_SCRIPT), "--design-id", did, "--n-train", str(n_train),
             "--source", "pilot", "--run-tag", "campaign80"],
            cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - start
    report_path = REPO_ROOT / f"Comparison/results/dataset_design_w90_001_s9_cross_testing_hparam_probe/probe_report__{did}__campaign80.json"
    result = {"rank": combo["rank"], "design_id": did, "returncode": completed.returncode, "elapsed_seconds": elapsed}
    if report_path.exists():
        result["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        result["error"] = f"no report written; see {log_path}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=7, help="Concurrent training jobs.")
    parser.add_argument("--limit", type=int, default=None, help="Debug: only run the first N combos.")
    args = parser.parse_args()

    combos = json.loads(COMBOS_PATH.read_text(encoding="utf-8"))["combos"]
    if args.limit:
        combos = combos[: args.limit]
    total = len(combos)
    selected_ids = {combo["w90_design_id"] for combo in combos}

    python_exe = str(REPO_ROOT / ".venv/bin/python")
    print(json.dumps({"n_combos": len(combos), "parallel": args.parallel}, sort_keys=True))

    results: list[dict] = []
    if SUMMARY_PATH.exists():
        try:
            results = [
                row for row in json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
                if row.get("returncode") == 0 and row.get("design_id") in selected_ids
            ]
        except (json.JSONDecodeError, OSError):
            results = []
    completed_ids = {row["design_id"] for row in results}
    combos = [combo for combo in combos if combo["w90_design_id"] not in completed_ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel, thread_name_prefix="w90-campaign") as executor:
        futures = {executor.submit(run_one, combo, python_exe): combo for combo in combos}
        for future in concurrent.futures.as_completed(futures):
            combo = futures[future]
            result = future.result()
            results.append(result)
            h_mae = result.get("report", {}).get("H_MAE_meV")
            print(json.dumps({"done": len(results), "of": total, "design_id": combo["w90_design_id"], "H_MAE_meV": h_mae, "rc": result["returncode"]}, sort_keys=True), flush=True)
            SUMMARY_PATH.write_text(json.dumps(sorted(results, key=lambda r: r["rank"]), indent=2, sort_keys=True), encoding="utf-8")

    n_ok = sum(1 for r in results if r["returncode"] == 0)
    print(json.dumps({"finished": True, "n_ok": n_ok, "n_total": len(results)}, sort_keys=True))
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
