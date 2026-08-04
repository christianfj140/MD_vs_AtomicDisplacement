#!/usr/bin/env python3
"""Assign the registry grid to train/validation/test by whole symmetry classes.

Splitting by row index would leak: registries in the same symmetry class are the
same physics up to a lattice operation, so a naive split can put a registry in
test whose twin is in train and measure nothing. Whole classes move together.

Classes containing a registry the MD training set already covers (AA, AB, BA) are
forced into train, because they are not held out in any meaningful sense. The
remaining classes are stratified by distance to the nearest covered registry so
that the held-out sets span the near-to-far range rather than clustering.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True,
                        help="Dataset whose material_provenance.json is copied (same basis).")
    parser.add_argument("--test-classes", type=int, default=4)
    parser.add_argument("--validation-classes", type=int, default=3)
    args = parser.parse_args()

    manifest_path = args.grid / "frozen_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]

    # The merger checks basis/pseudo hashes; they are byte-identical here, so the
    # reference provenance transfers unchanged.
    provenance = json.loads(
        (args.reference_dataset / "material_provenance.json").read_text(encoding="utf-8")
    )
    (args.grid / "material_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_class: dict[int, list[dict]] = {}
    for row in rows:
        by_class.setdefault(int(row["symmetry_class"]), []).append(row)

    covered = {c for c, items in by_class.items()
               if any(item["registry_seen_in_training"] for item in items)}
    free = sorted(
        (c for c in by_class if c not in covered),
        key=lambda c: float(np.mean([i["distance_to_nearest_seen_registry_ang"]
                                     for i in by_class[c]])),
    )
    held = args.test_classes + args.validation_classes
    if held >= len(free):
        raise RuntimeError(f"Only {len(free)} free classes; cannot hold out {held}.")
    # Stratify: pick evenly spaced classes along the distance ordering.
    picked = [free[i] for i in np.linspace(0, len(free) - 1, held).round().astype(int)]
    picked = list(dict.fromkeys(picked))
    test_classes = set(picked[0::2][: args.test_classes])
    validation_classes = set(picked[1::2][: args.validation_classes]) - test_classes

    counts = {"train": 0, "validation": 0, "test": 0}
    for cls, items in by_class.items():
        split = ("test" if cls in test_classes
                 else "validation" if cls in validation_classes else "train")
        for item in items:
            item["split"] = split
            counts[split] += 1

    for row in rows:
        old = Path(row["sample_dir"])
        new = args.grid / "splits" / row["split"] / old.name
        if old.resolve() != new.resolve():
            new.parent.mkdir(parents=True, exist_ok=True)
            if new.exists():
                shutil.rmtree(new)
            shutil.move(str(old), str(new))
        for key, suffix in (("sample_dir", ""), ("structure_path", "/RUN.fdf"),
                            ("metadata_path", "/metadata.json")):
            row[key] = str(new) + suffix
        row["reference_tshs_path"] = str(next(new.glob("*.TSHS")))

    manifest["split_policy"] = "whole symmetry classes; covered registries forced to train"
    manifest["held_out_classes"] = {"test": sorted(test_classes),
                                    "validation": sorted(validation_classes)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name in ("validation", "test"):
        empty = args.grid / "splits" / name
        if empty.is_dir() and not any(empty.iterdir()):
            empty.rmdir()
    print(f"clases: {len(by_class)} | test {sorted(test_classes)} | "
          f"validacion {sorted(validation_classes)}")
    print(f"muestras: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
