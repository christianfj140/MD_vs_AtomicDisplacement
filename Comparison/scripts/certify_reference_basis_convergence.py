#!/usr/bin/env python3
"""Aggregate the preregistered QZ cardinality/range evidence without recomputation."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
from artifact_signature import file_sha256  # noqa: E402

ROOT = REPO_ROOT / "Comparison/results/epc"
OUTPUT = ROOT / "reference_basis_convergence/reference_basis_convergence.json"
SOURCES = {
    "production_basis_connection": ROOT / "production_basis_connection/production_basis_connection.json",
    "qz_nested_identity_preflight": ROOT / "qz_nested_identity_preflight/qz_nested_identity_preflight.json",
    "qz_cardinality_C1_x": ROOT / "qz_cardinality_sentinel/qz_cardinality_sentinel.json",
    "qz_range_C1_x_and_cardinality_C1_z": ROOT / "qz_convergence_followup/qz_convergence_followup.json",
    "dense_k_audit": ROOT / "reference_basis_convergence/dense_k_audit.json",
}


def _verdict(reports: dict[str, dict]) -> str:
    return "PASS" if all(report.get("verdict") == "PASS" for report in reports.values()) else "NO_GO"


def main() -> int:
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}
    verdict = _verdict(reports)
    report = {
        "schema": "reference_basis_convergence_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict, "gate": "reference_basis_convergence",
        "sources": {name: {"path": str(SOURCES[name]), "sha256": file_sha256(SOURCES[name]),
                           "verdict": source["verdict"]} for name, source in reports.items()},
        "historical_results_preserved": ["standard_basis_cross_projection_v1=NO_GO",
                                         "reference_basis_convergence_raw=NO_GO",
                                         "reference_basis_convergence_covariant_DZP_to_TZP=NO_GO"],
        "evidence": "Nested QZ prefix identity plus final QZ cardinality for C1:x/C1:z and QZ diffuse-range at worst-case C1:x.",
        "claim_scope": "SIESTA covariant reference expressed in fixed PROD-SZ is stable against the preregistered final PAO cardinality/range increments; not Delta_out or full-KS.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference basis convergence: {verdict} -> {OUTPUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
