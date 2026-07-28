#!/usr/bin/env python3
"""Replay the locked N_min metrics with the 2000-replicate remediation protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    REPO_ROOT
    / "Comparison/results/dataset_size_minimum_audit_graphene_w90_reuse_10_1000_4seeds_20260612"
    / "dataset_size_minimum_summary.json"
)
OUTPUT = REPO_ROOT / "Comparison/results/audit_remediation_20260728/n_min_paper_audit"
METRICS = OUTPUT / "locked_metric_replay/summary/ranking/normalized_run_metrics.json"
PROTOCOL = REPO_ROOT / "Comparison/config/graphene_dataset_size_minimum_paper_threshold_protocol.json"
ANALYZER = REPO_ROOT / "Comparison/scripts/g2m_deeph_dataset_size_minimum.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = [
        {
            "model": row["method"],
            "config_id": row["config_id"],
            "seed": row.get("seed"),
            "n_total": row["dataset_size_total"],
            "n_train": row["dataset_size_train"],
            "h_mae_eV_mean": float(row["primary_metric_mev"]) / 1000.0,
            "source_run_root": row.get("source_run_root"),
            "source_metric_summary": str(SOURCE),
        }
        for row in source.get("normalized_rows") or []
    ]
    write_json(
        METRICS,
        {
            "schema": "locked_historical_metric_replay_v1",
            "source_summary": str(SOURCE),
            "source_summary_sha256": sha256(SOURCE),
            "physical_recalculation_performed": False,
            "rows": rows,
        },
    )
    analysis_dir = OUTPUT / "threshold_10meV"
    command = [
        sys.executable,
        str(ANALYZER),
        "--run-root",
        str(METRICS.parents[2]),
        "--output-dir",
        str(analysis_dir),
        "--primary-metric",
        "h_mae_eV_mean",
        "--threshold-mev",
        "10",
        "--threshold-preset-key",
        "h_mae_relaxed_10",
        "--threshold-is-user-defined",
        "false",
        "--threshold-protocol-file",
        str(PROTOCOL),
        "--relative-tolerance",
        "0.05",
        "--plateau-gain",
        "0.05",
        "--x-axis",
        "n_train",
        "--aggregation-mode",
        "mean_seeds_per_config",
        "--cost-basis",
        "protocol_total",
        "--claim-mode",
        "paper_candidate",
        "--fit-models",
        "linear,quadratic,inverse,inverse_square,power_law_floor",
        "--n-min-source",
        "fit",
        "--n-min-fit-model",
        "power_law_floor",
        "--moving-average-window",
        "3",
        "--bootstrap-replicates",
        "2000",
        "--bootstrap-seed",
        "12345",
        "--ci-level",
        "0.95",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    summary_path = analysis_dir / "dataset_size_minimum_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    replicate = summary.get("replicate_bootstrap") or {}
    hierarchy = summary.get("hierarchical_uncertainty") or {}
    sensitivity = summary.get("threshold_sensitivity") or {}
    execution_checks = {
        "command_returncode_zero": completed.returncode == 0,
        "bootstrap_requested_2000": replicate.get("replicates_requested") == 2000,
        "bootstrap_executed_2000": replicate.get("replicates_executed") == 2000,
        "requested_executed_match": replicate.get("requested_executed_match") is True,
        "hierarchical_replicates_2000": hierarchy.get("replicates") == 2000,
        "threshold_sensitivity_8_10_12": sensitivity.get("thresholds_mev") == [8.0, 10.0, 12.0],
    }
    blockers = list(summary.get("paper_level_blockers") or [])
    blockers.extend(
        [
            "source_metrics_are_historical_replay_not_new_physical_execution",
            "source_datasets_fail_current_strict_SCF_reference_revalidation",
        ]
    )
    if not all(execution_checks.values()):
        blockers.append("requested_N_min_remediation_execution_mismatch")
    audit = {
        "schema": "n_min_remediation_execution_audit_v1",
        "status": "BLOCKED_FAIL_CLOSED" if blockers else "PASS",
        "claim_allowed": not blockers,
        "source_summary": str(SOURCE),
        "source_summary_sha256": sha256(SOURCE),
        "locked_metrics": str(METRICS),
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "summary": str(summary_path),
        "execution_checks": execution_checks,
        "scientific_claim_status": summary.get("scientific_claim_status"),
        "N_min_nominal": summary.get("N_min_nominal"),
        "paper_level_blockers": sorted(set(str(item) for item in blockers)),
    }
    write_json(OUTPUT / "n_min_remediation_execution_audit.json", audit)
    return 0 if all(execution_checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
