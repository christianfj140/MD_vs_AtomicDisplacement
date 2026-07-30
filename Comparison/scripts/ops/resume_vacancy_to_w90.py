#!/usr/bin/env python3
"""Resume vacancy 5x5 -> w90 without retraining completed DeepH models."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "Comparison/config/cross_vacancy_to_w90_train_payload.json"
OUT = REPO / "Comparison/results/ml_vs_siesta_cross_structure_vacancy_to_w90"
RUNNER = REPO / "Comparison/scripts/run_cross_structure_sweep_payload.py"
PYTHON = REPO / ".venv/bin/python"


def pair_dir(pair: dict) -> Path:
    source = Path(pair["source"]).name
    target = Path(pair["target"]).name
    matches = sorted(OUT.glob(f"*__{source}__to__*__{target}"))
    return matches[0] if matches else OUT / f"missing__{source}__to__{target}"


def deeph_checkpoint(pair: dict) -> Path | None:
    checkpoints = sorted(pair_dir(pair).glob("training/deeph/**/deeph/train/best_state_dict.pkl"))
    return checkpoints[-1] if checkpoints else None


def payload(base: dict, model: str, pairs: list[dict], action: str) -> dict:
    result = copy.deepcopy(base)
    result["action"] = action
    result["models"] = [model]
    result["pairs"] = pairs
    result["sources"] = list(dict.fromkeys(pair["source"] for pair in pairs))
    result["targets"] = list(dict.fromkeys(pair["target"] for pair in pairs))
    performance = result["performance"]
    performance["cross_model_schedule"] = "single_model"
    performance["max_parallel_prediction_jobs"] = (
        performance["max_parallel_graph2mat_training_jobs"]
        if model == "graph2mat" and action == "train"
        else 1
    )
    return result


def run(body: dict, name: str, action: str, temp: Path) -> dict:
    if not body["pairs"]:
        return {"records": [], "permutations": [], "n_failed": 0}
    payload_path = temp / f"{name}.json"
    result_path = OUT / f"{name}_result.json"
    payload_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            str(PYTHON),
            "-u",
            str(RUNNER),
            str(payload_path),
            "--action",
            action,
            "--output-root",
            str(OUT),
            "--result-json",
            str(result_path),
        ],
        cwd=REPO,
        check=False,
    )
    if not result_path.is_file():
        raise RuntimeError(f"{name} failed with rc={completed.returncode} and produced no result")
    return json.loads(result_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    base = json.loads(BASE.read_text(encoding="utf-8"))
    completed = [(pair, deeph_checkpoint(pair)) for pair in base["pairs"]]
    done = [(pair, checkpoint) for pair, checkpoint in completed if checkpoint]
    missing = [pair for pair, checkpoint in completed if not checkpoint]
    print("DeepH completos:", [pair["size"] for pair, _ in done], flush=True)
    print("DeepH pendientes:", [pair["size"] for pair in missing], flush=True)
    print("Graph2Mat pendientes:", [pair["size"] for pair in base["pairs"]], flush=True)
    if args.plan:
        return 0

    with tempfile.TemporaryDirectory(prefix="vacancy-w90-resume-") as raw_temp:
        temp = Path(raw_temp)
        existing = payload(base, "deeph", [pair for pair, _ in done], "predict_metrics")
        existing["existing_artifacts"] = {
            Path(pair["source"]).name: {"deeph_save_dir": str(checkpoint.parent)}
            for pair, checkpoint in done
        }
        results = [
            run(existing, "resume_deeph_existing", "predict_metrics", temp),
            run(payload(base, "deeph", missing, "train"), "resume_deeph_missing", "train", temp),
            run(payload(base, "graph2mat", base["pairs"], "train"), "resume_graph2mat", "train", temp),
        ]

    records = {}
    for result in results:
        for record in result.get("records") or []:
            records[(record.get("payload_id"), record.get("model"), record.get("seed", 0))] = record
    summary = {
        "action": "train",
        "records": list(records.values()),
        "permutations": [
            permutation
            for result in results
            for permutation in result.get("permutations") or []
        ],
        "n_failed": sum(int(result.get("n_failed") or 0) for result in results),
        "resume": {
            "deeph_reused": [pair["size"] for pair, _ in done],
            "deeph_trained": [pair["size"] for pair in missing],
            "graph2mat_trained": [pair["size"] for pair in base["pairs"]],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Resumen combinado: {len(records)} registros", flush=True)
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
