#!/usr/bin/env python3
"""Re-grab invalid ``siesta_version`` fields in archived material_provenance.json.

Archived datasets recorded X11 noise ("Authorization required, but no
authorization protocol specified") as ``siesta_version`` because the old probe
accepted any non-empty first line. The real version survives inside
``siesta_build_info`` ("Version         : 5.4.2-11-g..."), so it can be
re-parsed in place without re-running SIESTA.

Idempotent. Dry-run by default; pass ``--apply`` to write. Each change is
logged as JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from benchmark_manifest import (  # noqa: E402
    extract_siesta_version_from_text,
    looks_like_siesta_version,
)

REGRAB_PROBE_STATUS = "regrabbed_from_build_info"


def regrab_provenance_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the regrab change for one provenance payload, or None.

    A change is proposed only when the recorded ``siesta_version`` does not
    validate as a version AND ``siesta_build_info`` yields one. The payload is
    modified in place when a change is returned.
    """
    old_version = payload.get("siesta_version")
    if looks_like_siesta_version(old_version):
        return None
    version = extract_siesta_version_from_text(payload.get("siesta_build_info"))
    if version is None:
        return None
    payload["siesta_version"] = version
    probe = payload.get("siesta_version_probe")
    if isinstance(probe, dict):
        probe["status"] = REGRAB_PROBE_STATUS
    else:
        payload["siesta_version_probe"] = {"status": REGRAB_PROBE_STATUS, "attempts": []}
    return {"old_version": old_version, "new_version": version}


def regrab_datasets(datasets_root: Path, *, apply: bool) -> list[dict[str, Any]]:
    """Scan ``material_provenance.json`` files and fix invalid versions.

    Returns one record per file inspected that needed (or received) a change.
    """
    changes: list[dict[str, Any]] = []
    for path in sorted(datasets_root.rglob("material_provenance.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            changes.append({"path": str(path), "status": "unreadable", "error": str(exc)})
            continue
        if not isinstance(payload, dict):
            continue
        change = regrab_provenance_payload(payload)
        if change is None:
            continue
        record = {"path": str(path), "status": "applied" if apply else "would_apply", **change}
        if apply:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        changes.append(record)
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=REPO_ROOT / "Comparison" / "datasets",
        help="Root scanned recursively for material_provenance.json files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the corrected files. Without this flag, dry-run only.",
    )
    args = parser.parse_args()

    changes = regrab_datasets(args.datasets_root, apply=args.apply)
    for record in changes:
        print(json.dumps(record, ensure_ascii=False))
    print(
        json.dumps(
            {
                "datasets_root": str(args.datasets_root),
                "apply": bool(args.apply),
                "n_changes": len([c for c in changes if c.get("new_version")]),
                "n_unreadable": len([c for c in changes if c.get("status") == "unreadable"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
