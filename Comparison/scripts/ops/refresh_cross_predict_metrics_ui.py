#!/usr/bin/env python3
"""Refresh cross-testing UI JSONs from completed common_summary files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ml_vs_siesta as mvs  # noqa: E402


ROOT = REPO_ROOT / "Comparison/results/ml_vs_siesta_cross_structure_sweep"
RUN_RE = re.compile(r"/(?P<payload>[^/]+)/training/g2m_deeph_[^/]+/common_metrics/summary/common_summary\.json$")


def main() -> int:
    records = []
    permutations = []
    for path in sorted(ROOT.glob("**/common_metrics/summary/common_summary.json")):
        match = RUN_RE.search(str(path))
        if not match:
            continue
        payload_id = match.group("payload")
        pair_root = path.parents[4]
        provenance = pair_root / "dataset/cross_structure_dataset_provenance.json"
        if not provenance.exists():
            continue
        prov = json.loads(provenance.read_text(encoding="utf-8"))
        source_root = Path(str(prov["source_dataset_root"]))
        target_root = Path(str(prov["target_dataset_root"]))
        source_id = mvs.cross_structure_sweep._dataset_id(source_root)  # type: ignore[attr-defined]
        target_id = mvs.cross_structure_sweep._dataset_id(target_root)  # type: ignore[attr-defined]
        source_n = mvs.cross_structure_sweep._source_train_count(source_root)  # type: ignore[attr-defined]
        permutations.append(
            {
                "payload_id": payload_id,
                "source_id": source_id,
                "target_id": target_id,
                "source_root": str(source_root),
                "target_root": str(target_root),
                "source_n_snapshots": source_n,
                "status": "evaluated",
                "output_root": str(pair_root),
            }
        )
        summary = json.loads(path.read_text(encoding="utf-8"))
        for row in summary.get("summary_rows") or []:
            mae = row.get("h_mae_eV_mean")
            model = row.get("method")
            if mae is None or model in (None, ""):
                continue
            records.append(
                {
                    "payload_id": payload_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_n_snapshots": source_n,
                    "model": str(model),
                    "seed": 0,
                    "h_mae_eV": float(mae),
                    "output_root": str(pair_root),
                    "partial_refresh": True,
                }
            )
    summary = {
        "schema": "ml_vs_siesta_cross_structure_sweep_summary_v1",
        "action": "predict_metrics",
        "partial_refresh": True,
        "n_permutations": len(permutations),
        "n_evaluated": len(permutations),
        "n_failed": 0,
        "permutations": permutations,
        "records": records,
    }
    (ROOT / "cross_structure_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mae_payload = mvs.aggregate_cross_structure_mae(records)
    (ROOT / "cross_structure_mae.json").write_text(json.dumps(mae_payload, indent=2), encoding="utf-8")
    print(f"refreshed records={len(records)} permutations={len(permutations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
