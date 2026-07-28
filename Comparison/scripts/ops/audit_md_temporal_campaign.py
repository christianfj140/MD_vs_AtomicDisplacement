#!/usr/bin/env python3
"""Audit the current MD campaign without mutating its historical outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for directory in (REPO_ROOT / "MD" / "scripts", REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from g2m_deeph_dataset_size_minimum import diagnose_dataset_temporal_metadata  # noqa: E402
from generate_md_dataset import build_md_temporal_evidence  # noqa: E402
from md_pipeline_config import load_pipeline_config, paths  # noqa: E402
from siesta_output_status import parse_siesta_output  # noqa: E402

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "audit_remediation_20260728"
    / "md_temporal_campaign_status.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "MD" / "pipeline_config.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)
    dataset_root = paths(config)["dataset_dir"]
    diagnostics = diagnose_dataset_temporal_metadata(dataset_root)
    evidence = build_md_temporal_evidence(config, diagnostics)
    siesta_status = parse_siesta_output(dataset_root / "RUN.out", dataset_root / "RUN.fdf")
    blockers = list(evidence["blockers"])
    if not siesta_status["valid"]:
        blockers.append(f"source_siesta_execution_invalid:{siesta_status['parser_status']}")
    payload = {
        "schema": "md_temporal_campaign_audit_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PASS" if evidence["paper_ready"] and siesta_status["valid"] else "BLOCKED_FAIL_CLOSED",
        "dataset_root": str(dataset_root),
        "source_siesta_status": siesta_status,
        "temporal_evidence": evidence,
        "paper_level_blockers": sorted(set(blockers)),
        "resume_command": (
            "PIPELINE_CONFIG_PATH=/absolute/path/to/a/new-paper-candidate-config.yaml "
            ".venv/bin/python MD/scripts/generate_md_dataset.py"
        ),
        "resume_requirements": [
            "Use a new dataset directory; do not overwrite the historical MD dataset.",
            "Execute and archive separate equilibration and production stages.",
            "Record canonical valid SIESTA status and executed/discarded step counts for both stages.",
            "Require stability checks, ACF, tau_int, N_eff and a block split derived from the predeclared gap rule.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
