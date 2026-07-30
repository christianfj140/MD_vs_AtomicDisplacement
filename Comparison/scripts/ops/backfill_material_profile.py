#!/usr/bin/env python
"""Backfill the `profile` field into dataset material_provenance.json files.

Commit 42ef983 (2026-07-28) added `profile` to ValidatedMaterialBundle.to_manifest_dict
and made `material_profile == "production"` a requirement for benchmark_ready. Every
dataset on disk predates that field, so all 170 of them carry no `profile` at all, read
back as None, and can no longer be materialized into a composite for training.

This does not weaken the gate: it resolves the profile from the material bundle that
actually produced each dataset (the same resolution the generator would do today) and
writes it into the provenance. A material whose bundle is `diagnostic` or `smoke` stays
that way and will still be rejected -- only the missing value is filled in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from material_presets import resolve_material_bundle  # noqa: E402


def resolve_profile(provenance: dict) -> tuple[str | None, str]:
    """Return (profile, how) resolved from the dataset's own recorded material."""
    preset = provenance.get("preset")
    if preset:
        try:
            bundle = resolve_material_bundle({"material": {"mode": "preset", "preset": preset}})
            return bundle.validated.bundle.profile, f"preset:{preset}"
        except Exception as exc:  # noqa: BLE001 - report, never guess a profile
            return None, f"preset_failed:{exc}"
    # Composite datasets record no preset; their label encodes the pair. They inherit the
    # profile of their sources, which the caller resolves separately.
    return None, "no_preset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write files (default: dry run).")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["Comparison/datasets/*/*/material_provenance.json"],
        help="Glob(s) of material_provenance.json to backfill.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.roots:
        paths.extend(sorted(REPO_ROOT.glob(pattern)))

    filled = skipped = failed = 0
    reasons: dict[str, int] = {}
    for path in paths:
        try:
            provenance = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  !! {path}: {exc}")
            failed += 1
            continue
        if provenance.get("profile"):
            skipped += 1
            continue
        profile, how = resolve_profile(provenance)
        reasons[how.split(":")[0]] = reasons.get(how.split(":")[0], 0) + 1
        if not profile:
            failed += 1
            continue
        provenance["profile"] = profile
        if args.apply:
            path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        filled += 1

    print(f"  rellenados: {filled}   ya tenian: {skipped}   sin resolver: {failed}")
    print(f"  via: {reasons}")
    if not args.apply and filled:
        print("\nDry run. Relanza con --apply para escribirlo.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
