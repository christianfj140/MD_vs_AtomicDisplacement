#!/usr/bin/env python
"""Backfill `positive_siesta_reference_provenance_v3` for datasets generated before it existed.

Commit 42ef983 (2026-07-28) rewrote choose_reference_matrix to demand a signed provenance
record binding the reference matrix, RUN.fdf, RUN.out, ORB_INDX and geometry hashes. Only
the derivative-reference path ever wrote one, so every MD dataset in the repo has zero
signatures and its references are rejected downstream.

HONEST BACKFILL -- read this before trusting the output.

The signature exists to be adversarial: tests/test_reference_provenance.py proves it
detects a mutated or swapped Hamiltonian. That power comes from signing *at the moment
SIESTA writes the files*. A signature minted now can only attest to what is on disk now,
so it cannot certify the original run -- it would otherwise be a self-issued certificate
of authenticity.

So every record written here carries `backfilled: true` plus the reason and timestamp.
Consumers keep working (the validator ignores unknown keys), but anyone auditing the
provenance can tell a generation-time signature from a retroactive one. Datasets
generated after the generator fix (write_reference_provenance in generate_md_dataset.py)
get the real thing and are never touched by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for directory in (REPO_ROOT / "shared", REPO_ROOT / "Comparison" / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from reference_provenance import build_positive_reference_provenance  # noqa: E402
from reference_selection import reference_candidates  # noqa: E402

BACKFILL_REASON = (
    "Firmado a posteriori: el dataset se genero antes de que existiera "
    "positive_siesta_reference_provenance_v3 (commit 42ef983, 2026-07-28). Los hashes "
    "corresponden a los ficheros en disco en el momento del backfill, NO certifican la "
    "ejecucion original de SIESTA."
)


def _load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def pick_reference(sample_dir: Path) -> Path | None:
    """TSHS wins over HSX, matching the pre-commit selection rule."""
    candidates = list(reference_candidates(sample_dir))
    tshs = [p for p in candidates if p.suffix == ".TSHS"]
    if len(tshs) == 1:
        return tshs[0]
    hsx = [p for p in candidates if p.suffix == ".HSX"]
    if not tshs and len(hsx) == 1:
        return hsx[0]
    return None


def backfill_dataset(dataset_root: Path, *, apply: bool) -> dict[str, int]:
    counts = {"written": 0, "already_signed": 0, "no_reference": 0, "ambiguous": 0}
    material = _load(dataset_root / "material_provenance.json")
    manifest = _load(dataset_root / "benchmark_dataset_manifest.json")
    frozen = _load(dataset_root / "frozen_split_manifest.json")
    split_hash = str(frozen.get("split_hash") or "")

    # splits/<split>/<n> holds symlinks into MD_steps/<n>, which is where the real
    # artifacts live. Cross-structure composites relink straight to MD_steps, so signing
    # only splits/ leaves those references unsigned -- sign both.
    sample_dirs: list[tuple[str, Path]] = []
    for split_dir in sorted((dataset_root / "splits").glob("*")):
        if split_dir.is_dir():
            sample_dirs.extend(
                (split_dir.name, p) for p in sorted(split_dir.iterdir()) if p.is_dir()
            )
    md_steps = dataset_root / "MD_steps"
    if md_steps.is_dir():
        split_of = {p.resolve(): split for split, p in sample_dirs}
        sample_dirs.extend(
            (split_of.get(p.resolve(), "train"), p)
            for p in sorted(md_steps.iterdir())
            if p.is_dir()
        )

    for split_name, sample_dir in sample_dirs:
        target = sample_dir / "siesta_reference_provenance.json"
        if target.exists():
            counts["already_signed"] += 1
            continue
        reference = pick_reference(sample_dir)
        if reference is None:
            counts["no_reference"] += 1
            continue
        provenance = build_positive_reference_provenance(
            sample_dir,
            reference,
            frozen_sample_id=f"md_{sample_dir.name}",
            split=split_name,
            frozen_split_hash=split_hash,
            basis_hashes=manifest.get("basis_hashes") or material.get("basis_file_sha256") or {},
            pseudopotential_hashes=(
                manifest.get("pseudopotential_hashes") or material.get("pseudopotential_sha256") or {}
            ),
            siesta_version=str(material.get("siesta_version") or ""),
            siesta_command=material.get("siesta_command_line") or "",
        )
        provenance["backfilled"] = True
        provenance["backfill_reason"] = BACKFILL_REASON
        provenance["backfilled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if apply:
            target.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        counts["written"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write files (default: dry run).")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["Comparison/datasets/graphene_5x5_vacancy_snapshot_scaling/*"],
        help="Glob(s) of dataset roots to sign.",
    )
    args = parser.parse_args()

    roots: list[Path] = []
    for pattern in args.datasets:
        roots.extend(sorted(p for p in REPO_ROOT.glob(pattern) if (p / "splits").is_dir()))

    total = {"written": 0, "already_signed": 0, "no_reference": 0, "ambiguous": 0}
    for root in roots:
        counts = backfill_dataset(root, apply=args.apply)
        for key, value in counts.items():
            total[key] += value
        print(f"  {root.name}: firmados={counts['written']} ya_firmados={counts['already_signed']} "
              f"sin_referencia={counts['no_reference']}")

    print(f"\n  TOTAL: {total}")
    if not args.apply and total["written"]:
        print("\nDry run. Relanza con --apply para escribirlo.")
        print("Los registros llevaran backfilled=true: firma retroactiva, no certifica la ejecucion original.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
