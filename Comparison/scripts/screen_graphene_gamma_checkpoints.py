#!/usr/bin/env python3
"""Screen existing Graph2Mat checkpoints against frozen graphene-Gamma evidence.

This is an exploratory, post-selection diagnostic. It never runs SIESTA and
never writes into the canonical C21 result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "Comparison/scripts/run_graphene_gamma_epc_paths.py"
CHECKPOINTS = ROOT / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints"
OUTPUT = ROOT / "Comparison/results/epc/checkpoint_screen_graphene_gamma_20260908"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoints() -> list[Path]:
    return sorted(
        [*CHECKPOINTS.glob("spectral-best-*.ckpt"), *CHECKPOINTS.glob("periodic/*.ckpt")],
        key=lambda path: (step(path), path.name),
    )


def step(path: Path) -> int:
    match = re.search(r"step[=-](\d+)", path.name)
    return int(match.group(1)) if match else -1


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_protocol(paths: list[Path]) -> dict:
    payload = {
        "schema": "graphene_gamma_checkpoint_screen_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "FROZEN_BEFORE_EXECUTION",
        "purpose": "diagnose whether GO4 model error is checkpoint-local or trajectory-wide",
        "ranking": [
            "quantitative_for_g descending",
            "worst_relative_subspace_error_on_result ascending",
            "tau_model ascending",
        ],
        "adequacy_threshold": 0.05,
        "limitations": [
            "exploratory post-selection screen; not a blind holdout",
            "cannot authorize fine-tuning or replace the canonical C21 verdict",
            "uses existing SIESTA artifacts read-only; no new DFT",
        ],
        "checkpoints": [
            {"path": str(path), "sha256": sha256(path), "step": step(path)} for path in paths
        ],
    }
    protocol_path = OUTPUT / "screen_protocol.json"
    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
        if frozen["checkpoints"] != payload["checkpoints"]:
            raise SystemExit("checkpoint set differs from the frozen screen protocol")
        return frozen
    write_json(protocol_path, payload)
    return payload


def metrics(summary: dict, checkpoint: Path) -> dict:
    tau = summary["tolerances"]["tau_model"]
    result = list(tau["result"]) + list(tau.get("result_graph2mat_frozen", []))
    validation = list(tau["validation"])
    value = float(tau["value"])
    worst = max(float(row["relative_subspace_error"]) for row in result)
    transfer = max(float(row["relative_subspace_error"]) for row in validation) <= value
    coverage = worst <= value
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "step": step(checkpoint),
        "tau_model": value,
        "worst_result_error": worst,
        "transfers_to_validation_split": transfer,
        "bound_covers_result": coverage,
        "quantitative_for_g": transfer and coverage and value <= 0.05,
    }


def plot(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted((row for row in rows if row["step"] >= 0), key=lambda row: row["step"])
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot([row["step"] for row in ordered], [row["tau_model"] for row in ordered],
              "o-", label=r"$\tau_{model}$")
    axis.plot([row["step"] for row in ordered], [row["worst_result_error"] for row in ordered],
              "s-", label="max result error")
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="quantitative threshold")
    axis.set_yscale("log")
    axis.set_xlabel("training step")
    axis.set_ylabel("relative subspace error")
    axis.set_title("Exploratory graphene Γ checkpoint screen")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "checkpoint_error_vs_training_step.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    paths = checkpoints()
    protocol = freeze_protocol(paths)
    print(f"frozen screen: {len(paths)} checkpoints -> {OUTPUT}", flush=True)
    if args.plan_only:
        return 0

    rows = []
    for index, checkpoint in enumerate(paths, 1):
        run_root = OUTPUT / "runs" / f"step_{step(checkpoint):05d}_{checkpoint.stem}"
        summary_path = run_root / "graphene_gamma_epc_summary.json"
        print(f"[{index}/{len(paths)}] {checkpoint.name}", flush=True)
        if not summary_path.exists():
            run_root.mkdir(parents=True, exist_ok=True)
            with (run_root / "screen.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, str(RUNNER), "--checkpoint", str(checkpoint),
                     "--result-root", str(run_root)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            if completed.returncode:
                print(f"  failed ({completed.returncode}); see {run_root / 'screen.log'}", flush=True)
                continue
        rows.append(metrics(json.loads(summary_path.read_text(encoding="utf-8")), checkpoint))
        write_json(OUTPUT / "checkpoint_screen.json", {
            "schema": protocol["schema"], "protocol": "screen_protocol.json", "results": rows
        })

    with (OUTPUT / "checkpoint_screen.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        plot(rows)
    print(f"complete: {len(rows)}/{len(paths)} checkpoints", flush=True)
    return 0 if len(rows) == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
