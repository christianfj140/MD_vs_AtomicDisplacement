#!/usr/bin/env python3
"""Audit geometry overlap between dataset/seed manifests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA = "geometry_independence_audit_v1"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest root is not an object: {path}")
    return payload


def geometry_hashes(payload: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for sample in payload.get("samples") or []:
        if isinstance(sample, dict) and sample.get("geometry_sha256"):
            hashes.add(str(sample["geometry_sha256"]))
    deterministic = payload.get("deterministic_hashes")
    if isinstance(deterministic, dict):
        for value in (deterministic.get("geometry_hashes") or {}).values():
            if value:
                hashes.add(str(value))
    return hashes


def build_audit(manifests: list[Path]) -> dict[str, Any]:
    sets = {str(path): geometry_hashes(read_json(path)) for path in manifests}
    pairs: list[dict[str, Any]] = []
    independent = len(sets) >= 2 and all(sets.values())
    names = sorted(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            intersection = sets[left] & sets[right]
            union = sets[left] | sets[right]
            jaccard = len(intersection) / len(union) if union else None
            pair = {
                "left": left,
                "right": right,
                "left_geometry_count": len(sets[left]),
                "right_geometry_count": len(sets[right]),
                "duplicate_geometry_count": len(intersection),
                "jaccard": jaccard,
                "exactly_identical": bool(sets[left]) and sets[left] == sets[right],
                "independent_geometry_sets": bool(sets[left] and sets[right] and not intersection),
            }
            pairs.append(pair)
            independent = independent and pair["independent_geometry_sets"]
    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifests": names,
        "geometry_counts": {name: len(sets[name]) for name in names},
        "pairs": pairs,
        "independent_replica_claim_allowed": independent,
        "experimental_unit": "geometry_family_or_trajectory_not_training_seed",
        "status": "independent" if independent else "overlap_or_insufficient_evidence",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_audit(args.manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pairs": len(report["pairs"])}))
    return 0 if report["independent_replica_claim_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
