#!/usr/bin/env python3
"""Run the real graphene Graph2Mat CUDA JVP preflight and persist its provenance."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import certify_siesta_dhsdr as go2  # noqa: E402
import run_graphene_gamma_epc_paths as gamma  # noqa: E402
from artifact_signature import file_sha256  # noqa: E402

REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/convergence/gauge_delta"
OUTPUT = REPO_ROOT / "Comparison/results/epc/preflight/graph2mat_jvp_runtime.json"


def main() -> int:
    campaign = go2.load_campaign(REFERENCE_ROOT)
    run_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = go2.read_tshs(
        campaign.run_dir(run_id), run_id, fermi_ev=campaign.fermi_ev(run_id)
    )
    side = gamma.ModelSide(gamma.DEFAULT_CHECKPOINT, campaign, equilibrium, OUTPUT.parent / "work")
    report = {
        "schema": "graph2mat_jvp_runtime_preflight_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(gamma.DEFAULT_CHECKPOINT),
        "checkpoint_sha256": file_sha256(gamma.DEFAULT_CHECKPOINT),
        "equilibrium_tshs": str(equilibrium.path),
        **side.backend.to_metadata(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Graph2Mat JVP preflight: {report['backend_preflight']} on {report['device']} -> {OUTPUT}")
    return 0 if report["backend_preflight"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
