#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S9-S3 -- SIESTA labeling for the pilot's coverage gap.

``dataset_design_w90_001_s9_s4_pilot_screening.py`` prunes a design entry as
"budget" when none of its generated geometries have a real SIESTA reference in
the S3 pool -- that is a labeling gap, not a time-budget decision, and it
cannot be closed by more GPU time. This script closes it: it takes the exact
pilot-scope design entries S9-S4 recorded as unlabelled
(``pilot_pruned.json``, entries whose ``detail`` names a missing SIESTA
reference), regenerates their geometries, keeps only the points that are not
already in the S3 pool, and runs SIESTA for those -- writing straight into
``Comparison/results/dataset_design_w90_001_s3/{structures,siesta_hamiltonians}``
so ``dataset_design_w90_001_s4_train_learning_curves.load_s3_samples`` and
every downstream S9 script picks them up with no further wiring.

Reuses, rather than reimplements:
  - ``w90_displacement_sampler_family`` / ``dataset_design_w90_001_s9_s3_design_generation``
    (S9-S2/S9-S3): geometry generation and hashing, identical to what produced
    ``design_manifest.json``.
  - ``run_hamiltonian_derivative_siesta_references.run_derivative_siesta_references``:
    the same idempotent SIESTA submission engine S3 and the pilot script use.
    Pointed at the S3 root itself, so the ~2660 already-labelled samples are a
    cheap existence check (``skip_if_exists``), not a repeated calculation.
  - ``run_w90_displacement_siesta_pilot``'s Hermiticity and orbital-dimension
    checks (``verify_hermiticity``, ``verify_orbital_dimension``), applied
    here only to the newly written samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import w90_displacement_sampler_family as sampler  # noqa: E402
from run_epc_siesta_reference import backend_preflight, canonical_ang_base_fdf  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    run_derivative_siesta_references,
)
from fdf_materialization import materialize_sample_fdf  # noqa: E402

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402
import dataset_design_w90_001_s9_s4_pilot_screening as s9s4  # noqa: E402
from run_w90_displacement_siesta_pilot import (  # noqa: E402
    DEFAULT_MATERIAL_FDF,
    HERMITICITY_ABS_TOLERANCE,
    verify_hermiticity,
    verify_orbital_dimension,
)

DEFAULT_S3_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s3"
DEFAULT_MANIFEST_PATH = s9s4.DEFAULT_MANIFEST_PATH
DEFAULT_PRUNED_PATH = s9s4.DEFAULT_OUTPUT_ROOT / "pilot_pruned.json"
DEFAULT_SIESTA_COMMAND = "/home/christian/bin/siesta"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s9_s3_labeling"

UNLABELLED_DETAIL_MARKER = "SIESTA reference"


class LabelingError(RuntimeError):
    pass


def unlabelled_pilot_designs(pruned_path: Path = DEFAULT_PRUNED_PATH) -> list[str]:
    """design_id of every pilot-scope entry S9-S4 pruned for lacking SIESTA labels."""
    pruned = json.loads(pruned_path.read_text(encoding="utf-8"))["pruned"]
    return sorted({
        entry["design_id"] for entry in pruned
        if UNLABELLED_DETAIL_MARKER in str(entry.get("detail", ""))
    })


def underlabelled_designs_at_domain(
    domain_ang: float, *, manifest_path: Path = DEFAULT_MANIFEST_PATH, tolerance_ang: float = 1e-9,
) -> list[str]:
    """design_id of every manifest cell at ``domain_ang`` whose SIESTA coverage is incomplete.

    Reads the manifest directly rather than ``pilot_pruned.json``: that file
    only ever lists cells a pilot RUN actually attempted, so a domain the
    coverage-matrix gate only just unblocked (e.g. a newly-enabled
    OOD_CONDITIONAL level) has no record there yet -- no pilot has tried it,
    so nothing marks it "missing." Checking the manifest's own
    n_reused_from_existing_siesta vs n_used works before the first pilot
    attempt, not only after.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["designs"]
    return sorted(
        entry["design_id"] for entry in manifest
        if abs(float(entry["r_train_max_ang"]) - domain_ang) < tolerance_ang
        and entry["n_used"] > 0
        and entry["n_reused_from_existing_siesta"] < entry["n_used"]
    )


def missing_configurations(
    design_ids: list[str],
    *,
    geometry: sampler.Geometry,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    s3_root: Path = DEFAULT_S3_ROOT,
) -> list[tuple[str, sampler.Configuration]]:
    """(sample_id, Configuration) for every point these designs need that S3 does not have.

    Deduplicated by geometry hash across all target designs, not per-design:
    two designs that generate the same point (e.g. adjacent density levels of
    the same family) must label it once.
    """
    manifest = {entry["design_id"]: entry for entry in json.loads(manifest_path.read_text(encoding="utf-8"))["designs"]}
    hash_to_sample = s9s4.build_s3_hash_index(s4.load_s3_samples(s3_root))

    seen: set[str] = set()
    entries: list[tuple[str, sampler.Configuration]] = []
    for design_id in design_ids:
        entry = manifest.get(design_id)
        if entry is None:
            continue
        family, dim, k = entry["family"], entry["dim"], int(entry["k"])
        r_train_max_ang = float(entry["r_train_max_ang"])
        resolution = entry["density_level"]
        seed_value = entry["seeds"]["sampler"]
        seed = seed_value if isinstance(seed_value, int) else 0
        configs = s3.generate_design(geometry, family, dim, k, r_train_max_ang, resolution, seed)
        for index, config in enumerate(configs):
            digest = s3.point_geometry_hash(geometry, config)
            if digest in hash_to_sample or digest in seen:
                continue
            seen.add(digest)
            sample_id = f"{design_id}__pt{index:04d}__{digest[:10]}"
            entries.append((sample_id, config))
    return entries


def materialize_structures(
    entries: list[tuple[str, sampler.Configuration]],
    *,
    structures_root: Path,
    canonical_base_fdf: Path,
    base_positions: list[tuple[float, float, float]],
) -> None:
    for sample_id, config in entries:
        displaced = [
            tuple(
                base_positions[atom_index][axis] + config.displacements_ang.get(atom_index, (0.0, 0.0, 0.0))[axis]
                for axis in range(3)
            )
            for atom_index in range(len(base_positions))
        ]
        out_dir = structures_root / sample_id
        materialize_sample_fdf(canonical_base_fdf, out_dir / "RUN.fdf", positions_ang=displaced, single_point=True)
        metadata = dict(config.metadata)
        metadata["sample_id"] = sample_id
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")


def run_labeling(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    s3_root: Path = DEFAULT_S3_ROOT,
    siesta_command: str = DEFAULT_SIESTA_COMMAND,
    workers: int = 16,
    max_samples: int | None = None,
    domain_ang: float | None = None,
) -> dict[str, Any]:
    from fdf_materialization import extract_fdf_structure

    output_root.mkdir(parents=True, exist_ok=True)
    canonical_base_fdf = s3_root / "_canonical_base.fdf"
    if not canonical_base_fdf.exists():
        canonical_ang_base_fdf(DEFAULT_MATERIAL_FDF, canonical_base_fdf)
    base_structure = extract_fdf_structure(canonical_base_fdf)
    base_positions = [tuple(position) for position in base_structure.positions_ang]

    geometry = sampler.load_graphene_primitive()
    design_ids = (
        underlabelled_designs_at_domain(domain_ang)
        if domain_ang is not None
        else unlabelled_pilot_designs()
    )
    if not design_ids:
        raise LabelingError("no pilot-scope design is recorded as SIESTA-unlabelled; nothing to do")

    entries = missing_configurations(design_ids, geometry=geometry, s3_root=s3_root)
    if max_samples is not None:
        entries = entries[: max(1, int(max_samples))]
    if not entries:
        raise LabelingError(
            f"{len(design_ids)} unlabelled design(s) recorded, but every geometry they generate "
            "already has an S3 reference -- nothing new to label"
        )

    # Materialize into an ISOLATED staging area, not s3_root/structures directly:
    # run_derivative_siesta_references' discover_structure_samples() scans every
    # directory under stencil_root/structures, skip-check included. Pointed at
    # s3_root (the whole accumulated pool -- tens of thousands of entries after a
    # night of labeling runs), every invocation re-pays an O(pool size) skip-check
    # scan on top of the real work, which is how a ~20 min job measured out at
    # ~13.5/min (~12h projected) instead of the ~80/min this same engine hit
    # earlier tonight against a much smaller pool. output_reference_root is set
    # explicitly so the SIESTA references still land in the shared S3 pool that
    # load_s3_samples() reads -- only the *scan* is isolated, not the result.
    structures_root = output_root / "structures"
    materialize_structures(
        entries, structures_root=structures_root, canonical_base_fdf=canonical_base_fdf, base_positions=base_positions,
    )

    preflight = backend_preflight(siesta_command)
    write_json(output_root / "backend_preflight.json", preflight)

    try:
        run_manifest = run_derivative_siesta_references(
            stencil_root=output_root,
            output_reference_root=s3_root / "siesta_hamiltonians",
            source_dataset_root=REPO_ROOT / "materials/graphene",
            siesta_command=siesta_command,
            workers=workers,
            diagnostic_only=True,
        )
    except DerivativeSiestaReferenceError as exc:
        raise LabelingError(str(exc)) from exc

    new_ids = {sample_id for sample_id, _config in entries}
    output_reference_root = Path(run_manifest["output_reference_root"])
    new_rows = [row for row in run_manifest["rows"] if row["sample_id"] in new_ids]
    usable_new_ids = [row["sample_id"] for row in new_rows if row["status"] in {"ok", "staged", "skipped_existing"}]
    failed_new_ids = [row["sample_id"] for row in new_rows if row["status"] not in {"ok", "staged", "skipped_existing"}]

    # load_s3_samples() reads structures from s3_root/structures, not from this
    # script's isolated staging area -- move (not copy) each usable structure
    # into the shared pool now that SIESTA has actually run on it. The
    # reference itself already landed in s3_root/siesta_hamiltonians directly
    # (output_reference_root was pointed there above).
    shared_structures_root = s3_root / "structures"
    shared_structures_root.mkdir(parents=True, exist_ok=True)
    for sid in usable_new_ids:
        source = structures_root / sid
        destination = shared_structures_root / sid
        if source.exists() and not destination.exists():
            source.rename(destination)

    hermiticity_rows = {sid: verify_hermiticity(output_reference_root / sid) for sid in usable_new_ids}
    orbital_rows = {sid: verify_orbital_dimension(output_reference_root / sid) for sid in usable_new_ids}
    hermiticity_failures = [sid for sid, row in hermiticity_rows.items() if row["status"] != "ok" or not row["pass"]]
    orbital_failures = [
        sid for sid, row in orbital_rows.items() if row["status"] != "ok" or not row["matches_expected"]
    ]

    report = {
        "task": "DATASET-DESIGN-W90-001-S9-S3-labeling",
        "target_design_ids": design_ids,
        "n_target_designs": len(design_ids),
        "n_new_geometries_requested": len(entries),
        "n_new_geometries_usable": len(usable_new_ids),
        "n_new_geometries_failed": len(failed_new_ids),
        "failed_sample_ids": failed_new_ids[:50],
        "siesta_command": siesta_command,
        "backend_preflight": preflight,
        "hermiticity_abs_tolerance_ev": HERMITICITY_ABS_TOLERANCE,
        "hermiticity_failure_count": len(hermiticity_failures),
        "hermiticity_failures": hermiticity_failures[:50],
        "orbital_dimension_failure_count": len(orbital_failures),
        "orbital_dimension_failures": orbital_failures[:50],
        "run_manifest_samples_total": run_manifest["samples_total"],
        "run_manifest_samples_ok": run_manifest["samples_ok"],
    }
    write_json(output_root / "labeling_report.json", report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--s3-root", type=Path, default=DEFAULT_S3_ROOT)
    parser.add_argument("--siesta-command", default=DEFAULT_SIESTA_COMMAND)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug/testing cap on new samples.")
    parser.add_argument(
        "--domain", type=float, default=None,
        help="Target a specific R_train_max (Ang) directly from the manifest instead of "
             "pilot_pruned.json -- needed for a domain no pilot run has ever attempted yet "
             "(e.g. a newly-enabled OOD_CONDITIONAL level), which pilot_pruned.json cannot "
             "know about.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    report = run_labeling(
        output_root=args.output_root,
        s3_root=args.s3_root,
        siesta_command=args.siesta_command,
        workers=args.workers,
        max_samples=args.max_samples,
        domain_ang=args.domain,
    )
    print(json.dumps({
        "n_new_geometries_usable": report["n_new_geometries_usable"],
        "n_new_geometries_failed": report["n_new_geometries_failed"],
        "hermiticity_failure_count": report["hermiticity_failure_count"],
        "orbital_dimension_failure_count": report["orbital_dimension_failure_count"],
    }, sort_keys=True))
    # Hermiticity is diagnostic, not a gate: run_w90_displacement_siesta_pilot.py's own
    # hermiticity_summary documents a fixed ~1e-10-to-few-1e-9 floor from this SIESTA
    # build's TSHS write (MeshCutoff 600 Ry) that a meaningful fraction of samples miss at
    # the strict 1e-10 tolerance regardless of displacement -- the historical S3 pool sits
    # at 38.5% "failing" that tolerance, and it's used throughout training unfiltered. A
    # geometry with an unparseable/missing orbital dimension is a real defect; one with a
    # measured-but-nonzero Hermiticity deviation is not.
    ok = report["n_new_geometries_failed"] == 0 and report["orbital_dimension_failure_count"] == 0
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
