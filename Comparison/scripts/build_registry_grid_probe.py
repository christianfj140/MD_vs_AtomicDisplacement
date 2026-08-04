#!/usr/bin/env python3
"""Build a lateral-registry probe set for pure bilayer graphene.

Purpose: measure what the trained model does on the interlayer registries that
actually fill a moire cell, without retraining anything. The AA/AB/BA training
set samples only three points of a continuous 2-D registry space; a moire cell
sweeps all of it. This builds a fractional grid of relative in-plane shifts of
the top layer, runs one static SIESTA SCF per registry, and assembles a dataset
directory that ``evaluate_checkpoint_spectral_metrics.py`` can score directly.

The grid deliberately includes (0,0) = AA as an internal control (a registry the
model has seen) and avoids (1/3,2/3) and (2/3,1/3), which are AB and BA, so every
other point is a registry the model has never been trained on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FDF = REPO_ROOT / "materials/bilayer_graphene_AA/RUN.fdf"
PSEUDO = REPO_ROOT / "materials/graphene_common/pseudos/C.psf"
BASIS = REPO_ROOT / "materials/graphene_common/basis/C.ion.xml"
SIESTA = Path("/home/christian/bin/siesta")
SYSTEM_LABEL = "bilayer_graphene_AA"
TOP_LAYER_Z = 11.0  # atoms above this z belong to the top layer
# AA is (0,0); AB and BA sit at these fractional shifts and are excluded.
SEEN_REGISTRIES = ((1 / 3, 2 / 3), (2 / 3, 1 / 3))
MIN_FREE_DISK_PERCENT = 12.0  # same floor the solver aborts at


def safe_output_root(path: Path) -> Path:
    """Refuse to write (and later delete) anything outside the datasets tree."""
    resolved = path.expanduser().resolve(strict=False)
    allowed = ((REPO_ROOT / "Comparison" / "datasets").resolve(),
               Path(tempfile.gettempdir()).resolve())
    if not any(root in resolved.parents for root in allowed):
        raise RuntimeError(
            f"Refusing output root {resolved}; use a child of Comparison/datasets "
            f"or {tempfile.gettempdir()}."
        )
    return resolved


def check_disk(root: Path) -> None:
    usage = shutil.disk_usage(root if root.exists() else root.parent)
    free_percent = 100.0 * usage.free / usage.total
    if free_percent < MIN_FREE_DISK_PERCENT:
        raise RuntimeError(
            f"Only {free_percent:.1f}% free on {root}; refusing to start "
            f"(floor {MIN_FREE_DISK_PERCENT}%)."
        )


def read_cell_and_atoms(fdf: Path) -> tuple[np.ndarray, list[list[str]], list[str]]:
    lines = fdf.read_text().splitlines()
    cell, atoms, block = [], [], None
    for line in lines:
        low = line.strip().lower()
        if low.startswith("%block latticevectors"):
            block = "cell"
            continue
        if low.startswith("%block atomiccoordinates"):
            block = "atoms"
            continue
        if low.startswith("%endblock"):
            block = None
            continue
        if block == "cell" and line.split():
            cell.append([float(v) for v in line.split()[:3]])
        elif block == "atoms" and line.split():
            atoms.append(line.split())
    return np.array(cell), atoms, lines


def write_shifted_fdf(destination: Path, shift_frac: tuple[float, float]) -> None:
    cell, atoms, lines = read_cell_and_atoms(SOURCE_FDF)
    shift = shift_frac[0] * cell[0][:2] + shift_frac[1] * cell[1][:2]
    out, block = [], None
    for line in lines:
        low = line.strip().lower()
        if low.startswith("%block atomiccoordinates"):
            block = "atoms"
            out.append(line)
            continue
        if low.startswith("%endblock atomiccoordinates"):
            block = None
            out.append(line)
            continue
        if block == "atoms" and line.split():
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            if z > TOP_LAYER_Z:
                x, y = x + shift[0], y + shift[1]
            out.append(f" {x:.10f}  {y:.10f}  {z:.10f}  {parts[3]}  # C")
        else:
            out.append(line)
    destination.write_text("\n".join(out) + "\n")


def run_siesta(sample_dir: Path) -> tuple[Path, int, float]:
    import time

    started = time.time()
    with (sample_dir / "RUN.fdf").open() as stdin, (sample_dir / "RUN.out").open("w") as stdout:
        completed = subprocess.run(
            [str(SIESTA)], cwd=sample_dir, stdin=stdin, stdout=stdout,
            stderr=subprocess.STDOUT, check=False,
        )
    return sample_dir, completed.returncode, time.time() - started


def registry_distance_ang(point: tuple[float, float], cell: np.ndarray) -> float:
    """Minimum-image distance to the nearest registry present in AA/AB/BA training.

    Must be measured in Cartesian angstrom, not as a Euclidean distance on the
    fractional coordinates: the lattice is hexagonal, so the fractional metric is
    not the identity and ranks the registries wrongly.
    """
    import itertools

    seen = [np.array([0.0, 0.0]), np.array([1 / 3, 2 / 3]), np.array([2 / 3, 1 / 3])]
    best = float("inf")
    for target, (n1, n2) in itertools.product(seen, itertools.product((-1, 0, 1), repeat=2)):
        delta = np.asarray(point) - target + np.array([n1, n2])
        best = min(best, float(np.linalg.norm(delta[0] * cell[0][:2] + delta[1] * cell[1][:2])))
    return best


def grid_points(divisions: int) -> list[tuple[float, float]]:
    """Every point of the fractional grid, seen registries included as controls."""
    return [(i / divisions, j / divisions) for i in range(divisions) for j in range(divisions)]


def symmetry_fingerprint(point: tuple[float, float], cell: np.ndarray,
                         n_neighbours: int = 36, decimals: int = 2) -> tuple:
    """A label equal for registries related by a lattice symmetry.

    Splitting train/test by raw grid index would leak: many registries are mirror
    or rotation images of each other and carry identical physics, so a naive split
    can put a registry in test whose symmetry twin is in train and measure nothing.
    The sorted interlayer C-C distances are invariant under those operations and
    need no explicit enumeration of the point group.

    Uses the ``n_neighbours`` shortest distances rather than a distance cutoff on
    purpose: a cutoff has a boundary, and a pair sitting on it is admitted for one
    registry and rejected for its symmetry twin, which silently breaks the
    invariance the whole fingerprint exists to provide.
    """
    bottom = np.array([[0.0, 0.0], [0.0, 1.4318286667]])
    interlayer_z = 3.35
    shift = point[0] * cell[0][:2] + point[1] * cell[1][:2]
    top = bottom + shift
    reach = 4
    distances = []
    for n1 in range(-reach, reach + 1):
        for n2 in range(-reach, reach + 1):
            offset = n1 * cell[0][:2] + n2 * cell[1][:2]
            for b in bottom:
                for t in top:
                    planar = np.linalg.norm(t + offset - b)
                    distances.append(float(np.hypot(planar, interlayer_z)))
    distances.sort()
    return tuple(round(d, decimals) for d in distances[:n_neighbours])


def grid_points(divisions: int) -> list[tuple[float, float]]:
    """Every point of the fractional grid, seen registries included as controls."""
    return [(i / divisions, j / divisions) for i in range(divisions) for j in range(divisions)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--divisions", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    points = grid_points(args.divisions)
    root = safe_output_root(args.output_root)
    check_disk(root)
    (root / "material_basis").mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASIS, root / "material_basis/C.ion.xml")
    shutil.copy2(PSEUDO, root / "C.psf")

    sample_dirs, pending = [], []
    for index, point in enumerate(points):
        sample_dir = root / "splits" / args.split / str(index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_shifted_fdf(sample_dir / "RUN.fdf", point)
        shutil.copy2(PSEUDO, sample_dir / "C.psf")
        shutil.copy2(BASIS, sample_dir / "C.ion.xml")
        sample_dirs.append((index, point, sample_dir))
        # Resume: a registry whose TSHS is already there is not recomputed.
        if not (sample_dir / f"{SYSTEM_LABEL}.TSHS").is_file():
            pending.append(sample_dir)

    reused = len(sample_dirs) - len(pending)
    print(f"{len(points)} registros ({args.divisions}x{args.divisions}); "
          f"{reused} reutilizados, {len(pending)} por calcular con {args.jobs} en paralelo")
    if pending:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_siesta, pending))
        failed = [str(d) for d, code, _ in results if code != 0]
        if failed:
            raise RuntimeError(f"SIESTA failed for: {failed}")
        elapsed = [t for _, _, t in results]
        print(f"SIESTA OK: {len(results)} SCF, {min(elapsed):.1f}-{max(elapsed):.1f} s cada uno, "
              f"pared {max(elapsed):.1f} s")

    cell, _atoms, _lines = read_cell_and_atoms(SOURCE_FDF)
    fingerprints = {symmetry_fingerprint(p, cell) for _, p, _ in sample_dirs}
    classes = {fp: index for index, fp in enumerate(sorted(fingerprints))}
    print(f"{len(classes)} clases de simetria distintas entre {len(sample_dirs)} registros")
    rows = []
    for index, point, sample_dir in sample_dirs:
        tshs = sample_dir / f"{SYSTEM_LABEL}.TSHS"
        if not tshs.is_file():
            raise RuntimeError(f"{sample_dir}: SIESTA produced no TSHS")
        seen = any(np.allclose(point, r, atol=1e-6)
                   for r in ((0.0, 0.0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)))
        distance = registry_distance_ang(point, cell)
        metadata = {
            "registry_fractional": list(point),
            "registry_seen_in_training": bool(seen),
            "distance_to_nearest_seen_registry_ang": distance,
            "symmetry_class": classes[symmetry_fingerprint(point, cell)],
            "note": "training registry (control)" if seen
                    else "registry absent from AA/AB/BA training set",
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append({
            "sample_id": f"reg_{index:02d}_{point[0]:.3f}_{point[1]:.3f}",
            "source_sample_id": f"reg_{index:02d}",
            "split": args.split,
            "valid": True,
            "status": "ok",
            "method": "static_registry_grid",
            "sample_dir": str(sample_dir),
            "structure_path": str(sample_dir / "RUN.fdf"),
            "metadata_path": str(sample_dir / "metadata.json"),
            "reference_tshs_path": str(tshs),
            "registry_fractional": list(point),
            "registry_seen_in_training": bool(seen),
            "distance_to_nearest_seen_registry_ang": distance,
            "symmetry_class": classes[symmetry_fingerprint(point, cell)],
        })
    manifest = {
        "dataset_kind": "lateral_registry_probe",
        "symmetry_class_count": len(classes),
        "split_guidance": "split by symmetry_class, never by row index: registries "
                          "sharing a class are physically equivalent and would leak",
        "purpose": "measure frontier-state error on registries absent from AA/AB/BA training",
        "divisions": args.divisions,
        "excluded_because_seen": [list(p) for p in SEEN_REGISTRIES],
        "rows": rows,
    }
    (root / "frozen_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"dataset escrito en {root} ({len(rows)} muestras, split={args.split})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
