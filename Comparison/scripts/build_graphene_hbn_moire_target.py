#!/usr/bin/env python3
"""Build a test-only twisted-bilayer graphene/hBN moire target with SIESTA refs.

Geometry: start from a single graphene/hBN stacking cell (4 atoms: 2 C graphene
layer + B + N hBN layer, both layers on the same in-plane hexagonal lattice),
then build the standard periodic commensurate supercell: layer 1 uses the
``(m,n)`` basis and layer 2 the ``(n,m)`` basis. Species, PAO basis and
pseudopotentials are byte-for-byte identical to the flat stackings, so the
cross-sweep planner accepts the bilayer->moire pair.

PHYSICS CAVEAT (documented, not hidden): graphene and hBN are incommensurate
(~1.8% for native 2.46-A graphene). This builder does NOT resolve the true incommensurate
moire; it applies a rigid commensurate-angle twist of the hBN layer on the
*shared* graphene lattice, which imposes an effective in-plane strain on hBN.
The applied angle and the implied strain are recorded in
``material_provenance.json`` under ``moire``. This is a smoke-scale surrogate
target for transfer testing, not a paper-ready incommensurate moire.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPT_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_manifest import file_sha256, write_benchmark_manifests  # noqa: E402
from fdf_materialization import extract_fdf_structure, materialize_fdf_text  # noqa: E402
from joint_artifact_contract import (  # noqa: E402
    G2M_DEEPH_BENCHMARK_PROFILE,
    validate_dataset,
    validate_snapshot,
)
from material_presets import resolve_material_bundle  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    run_siesta,
    stage_required_pseudopotentials,
)


SYSTEM_LABEL = "graphene_hBN_moire"
DEFAULT_STACKING = "graphene_hBN_AA"
COMMON_MATERIAL_ROOT = REPO_ROOT / "materials" / "graphene_hBN_common"
STATIC_DROP_PREFIXES = ("md.",)
STATIC_DROP_KEYS = {"writemdhistory", "lua.script"}
# hBN layer is the fractional-z upper sublayer in the stacking fdf (~0.5675 > 0.5).
HBN_Z_THRESHOLD = 0.5
HBN_NATIVE_LATTICE_ANG = 2.504
HBN_NATIVE_LATTICE_REFERENCE = "https://doi.org/10.1038/s42005-020-0335-1"
DEFAULT_MIN_ATOM_DISTANCE_ANG = 1.2


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    allowed = ((REPO_ROOT / "Comparison" / "datasets").resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(root in resolved.parents for root in allowed):
        raise RuntimeError(
            f"Refusing output root {resolved}; use a child of Comparison/datasets or {tempfile.gettempdir()}."
        )
    return resolved


def _static_fdf(text: str) -> str:
    lines = []
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        key = clean.split(None, 1)[0].lower() if clean else ""
        if key in STATIC_DROP_KEYS or any(key.startswith(prefix) for prefix in STATIC_DROP_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def commensurate_angle_degrees(m: int, n: int) -> float:
    """Standard hexagonal commensurate twist angle for coprime (m, n)."""
    if m <= 0 or n <= 0 or m == n:
        raise RuntimeError(f"Approximant (m, n) must be positive with m != n, got ({m}, {n}).")
    numerator = m * m + 4 * m * n + n * n
    denominator = 2 * (m * m + m * n + n * n)
    return math.degrees(math.acos(numerator / denominator))


def _rescale_inplane_kgrid(text: str, linear_scale: float) -> str:
    """Scale the two in-plane MP counts by cell length (min 1); keep k_z."""
    lines = text.splitlines()
    out: list[str] = []
    inblock = False
    row = 0
    for line in lines:
        low = line.split("#", 1)[0].strip().lower()
        if low.startswith("%block kgrid_monkhorst_pack"):
            inblock = True
            row = 0
            out.append(line)
            continue
        if low.startswith("%endblock kgrid_monkhorst_pack"):
            inblock = False
            out.append(line)
            continue
        if inblock and low:
            parts = line.split()
            if len(parts) >= 4 and row < 2:  # only the two in-plane rows
                parts[row] = str(max(1, round(int(parts[row]) / linear_scale)))
                out.append("  " + "  ".join(parts))
                row += 1
                continue
            row += 1
        out.append(line)
    return "\n".join(out)


def minimum_periodic_distance(positions: np.ndarray, lattice: np.ndarray) -> float:
    """Return the minimum pair distance over 3x3 in-plane periodic images."""
    best = math.inf
    shifts = [i * lattice[0] + j * lattice[1] for i in (-1, 0, 1) for j in (-1, 0, 1)]
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            delta = positions[right] - positions[left]
            best = min(best, *(float(np.linalg.norm(delta + shift)) for shift in shifts))
    return best


def validate_minimum_atom_distance(
    positions: np.ndarray,
    lattice: np.ndarray,
    minimum_distance_ang: float = DEFAULT_MIN_ATOM_DISTANCE_ANG,
) -> float:
    """Abort on overlapping atoms and return the measured periodic distance."""
    if minimum_distance_ang <= 0:
        raise RuntimeError(f"--min-atom-distance must be positive, got {minimum_distance_ang}.")
    distance = minimum_periodic_distance(np.asarray(positions, dtype=float), np.asarray(lattice, dtype=float))
    if distance < minimum_distance_ang:
        raise RuntimeError(
            f"Invalid moire geometry: minimum periodic atom distance {distance:.6f} Ang "
            f"is below --min-atom-distance {minimum_distance_ang:.6f} Ang."
        )
    return distance


def _layer_positions(
    atoms: list[Any],
    primitive_lattice: np.ndarray,
    super_lattice: np.ndarray,
    rotation: np.ndarray,
    expected: int,
) -> tuple[list[list[float]], list[int]]:
    """Materialize one periodic layer inside a common commensurate cell."""
    inverse = np.linalg.inv(super_lattice)
    bound = expected + 1
    positions: list[list[float]] = []
    species: list[int] = []
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            shift = i * primitive_lattice[0] + j * primitive_lattice[1]
            for atom in atoms:
                cart = np.asarray(atom.position_ang, dtype=float) + shift
                cart[:2] = cart[:2] @ rotation
                frac = cart @ inverse
                if np.all(frac[:2] >= -1e-9) and np.all(frac[:2] < 1.0 - 1e-9):
                    frac[:2] %= 1.0
                    positions.append((frac @ super_lattice).tolist())
                    species.append(atom.species_index)
    if len(positions) != 2 * expected:
        raise RuntimeError(f"Commensurate layer produced {len(positions)} atoms, expected {2 * expected}.")
    return positions, species


def moire_geometry(
    stacking_fdf: Path,
    *,
    approximant: int,
    m: int,
    n: int,
    min_atom_distance: float = DEFAULT_MIN_ATOM_DISTANCE_ANG,
) -> tuple[str, dict[str, Any]]:
    """Return a static moire FDF plus geometry metadata."""
    structure = extract_fdf_structure(stacking_fdf, structure_type="crystal")
    if structure.atom_count != 4:
        raise RuntimeError(f"Expected a 4-atom stacking cell in {stacking_fdf}, found {structure.atom_count}.")
    if approximant != 2:
        raise RuntimeError("--approximant is retained for CLI compatibility and must be 2; cell size comes from (m,n).")

    lattice = np.asarray(structure.lattice_vectors_ang, dtype=float)
    a1, a2, a3 = lattice
    inv_lattice = np.linalg.inv(lattice)
    twist_deg = commensurate_angle_degrees(m, n)
    theta = math.radians(twist_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=float)
    index = m * m + m * n + n * n
    layer1_matrix = np.array([[m + n, n], [m, m + n]], dtype=int)
    layer2_matrix = np.array([[m + n, m], [n, m + n]], dtype=int)
    super_lattice = np.array([*(layer1_matrix @ lattice[:2]), a3], dtype=float)
    if not np.allclose((layer2_matrix @ lattice[:2])[:, :2] @ rotation, super_lattice[:2, :2]):
        raise RuntimeError(f"({m},{n}) does not materialize the requested commensurate rotation.")

    graphene = [atom for atom in structure.atoms if float((np.asarray(atom.position_ang) @ inv_lattice)[2]) <= HBN_Z_THRESHOLD]
    hbn = [atom for atom in structure.atoms if float((np.asarray(atom.position_ang) @ inv_lattice)[2]) > HBN_Z_THRESHOLD]
    positions, species = _layer_positions(graphene, lattice, super_lattice, np.eye(2), index)
    hbn_positions, hbn_species = _layer_positions(hbn, lattice, super_lattice, rotation, index)
    positions += hbn_positions
    species += hbn_species
    expected = 4 * index
    if len(positions) != expected:
        raise RuntimeError(f"Moire geometry produced {len(positions)} atoms, expected {expected}.")
    dist_min = validate_minimum_atom_distance(np.asarray(positions), super_lattice, min_atom_distance)

    text = materialize_fdf_text(
        stacking_fdf.read_text(encoding="utf-8", errors="ignore"),
        structure,
        positions_ang=positions,
        atom_species=species,
        lattice_vectors_ang=super_lattice.tolist(),
        system_label=SYSTEM_LABEL,
        system_name=SYSTEM_LABEL,
    )
    # The commensurate cell is sqrt(index) larger in-plane, so its
    # Monkhorst-Pack mesh is scaled to keep the same k-point density
    # as the flat stacking. Otherwise the cross-sweep planner rejects the pair
    # for differing k-density (integer MP counts may differ, density may not).
    text = _rescale_inplane_kgrid(text, math.sqrt(index))
    actual_lattice_ang = float(np.linalg.norm(a1[:2]))
    lattice_mismatch_percent = (HBN_NATIVE_LATTICE_ANG / actual_lattice_ang - 1.0) * 100.0
    hbn_strain_percent = (actual_lattice_ang / HBN_NATIVE_LATTICE_ANG - 1.0) * 100.0
    metadata = {
        "structure_type": "crystal",
        "source_stacking": stacking_fdf.parent.name,
        "approximant": approximant,
        "commensurate_index": [m, n],
        "commensurate_cell_index": index,
        "layer1_supercell_matrix": layer1_matrix.tolist(),
        "layer2_supercell_matrix": layer2_matrix.tolist(),
        "twist_angle_deg": twist_deg,
        "materialized_twist_angle_deg": math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])),
        "num_atoms": len(positions),
        "twisted_sublayer": "hBN",
        "geometry_inplane_lattice_ang": actual_lattice_ang,
        "native_hBN_lattice_ang": HBN_NATIVE_LATTICE_ANG,
        "native_hBN_lattice_reference": HBN_NATIVE_LATTICE_REFERENCE,
        "graphene_hBN_lattice_mismatch_percent": lattice_mismatch_percent,
        "effective_hBN_strain_percent": hbn_strain_percent,
        "minimum_periodic_atom_distance_ang": dist_min,
        "minimum_atom_distance_threshold_ang": min_atom_distance,
        "approximation": (
            "standard periodic (m,n)/(n,m) commensurate twist; both layers use the shared graphene lattice, "
            "so hBN is biaxially compressed relative to its native 2.504-A lattice; "
            "true incommensurate moire is NOT resolved"
        ),
        "system_label": SYSTEM_LABEL,
        "relaxed": False,
        "spin_polarized": False,
    }
    return _static_fdf(text), metadata


def _copy_basis(output_root: Path) -> None:
    basis_dir = output_root / "material_basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    for basis in (COMMON_MATERIAL_ROOT / "basis").glob("*.ion.xml"):
        shutil.copy2(basis, basis_dir / basis.name)
    for pattern in ("*.psf", "*.psml"):
        for pseudo in (COMMON_MATERIAL_ROOT / "pseudos").glob(pattern):
            shutil.copy2(pseudo, output_root / pseudo.name)


def _write_test_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    path = output_root / "splits" / "test_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "method", "source_run", "source_sample_id", "structure_path",
        "hamiltonian_path", "run_out_path", "metadata_path", "valid", "split", "status", "sample_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _siesta_version(run_out: Path) -> str | None:
    if not run_out.is_file():
        return None
    for line in run_out.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().lower().startswith("version"):
            _, _, value = line.partition(":")
            if value.strip():
                return value.strip()
    return None


def _material_provenance(
    output_root: Path,
    stacking_fdf: Path,
    stacking_preset: str,
    geometry: dict[str, Any],
    siesta_command: str,
    first_sample: Path,
) -> dict[str, Any]:
    validated = resolve_material_bundle({"material": {"preset": stacking_preset}}).validated
    run_out = first_sample / "RUN.out"
    return {
        "label": SYSTEM_LABEL,
        "preset": SYSTEM_LABEL,
        "material_source": "twisted_bilayer_graphene_hBN_static_siesta",
        "fdf": str(stacking_fdf),
        "fdf_sha256": file_sha256(stacking_fdf),
        "structure_type": "crystal",
        "basis_file_sha256": validated.basis_file_sha256,
        "pseudopotential_sha256": validated.pseudopotential_sha256,
        "pseudopotentials": {k: str(v) for k, v in validated.pseudopotentials.items()},
        "species": [sp.to_dict() for sp in validated.species],
        "siesta_command_line": siesta_command,
        "siesta_version": _siesta_version(run_out),
        "siesta_stdout_path": str(run_out),
        "run_out_path": str(run_out),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "moire": geometry,
    }


def _prepare_geometry(
    stacking_fdf: Path,
    destination: Path,
    *,
    approximant: int,
    m: int,
    n: int,
    min_atom_distance: float,
) -> dict[str, Any]:
    text, metadata = moire_geometry(
        stacking_fdf, approximant=approximant, m=m, n=n, min_atom_distance=min_atom_distance
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "RUN.fdf").write_text(text, encoding="utf-8")
    _write_json(destination / "metadata.json", metadata)
    parsed = extract_fdf_structure(destination / "RUN.fdf", structure_type="crystal")
    if parsed.atom_count != metadata["num_atoms"]:
        raise RuntimeError(f"Moire geometry validation failed for {destination}: {parsed.atom_count} atoms.")
    validate_minimum_atom_distance(
        np.asarray([atom.position_ang for atom in parsed.atoms]),
        np.asarray(parsed.lattice_vectors_ang),
        min_atom_distance,
    )
    return metadata


def dry_run_plan(
    stacking_fdf: Path,
    *,
    approximant: int,
    m: int,
    n: int,
    limit: int,
    min_atom_distance: float = DEFAULT_MIN_ATOM_DISTANCE_ANG,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="moire_dry_run_") as tmp:
        metadata = _prepare_geometry(
            stacking_fdf,
            Path(tmp) / "0",
            approximant=approximant,
            m=m,
            n=n,
            min_atom_distance=min_atom_distance,
        )
    return {
        "dry_run": True,
        "stacking_fdf": str(stacking_fdf),
        "approximant": approximant,
        "commensurate_index": [m, n],
        "twist_angle_deg": metadata["twist_angle_deg"],
        "num_atoms": metadata["num_atoms"],
        "minimum_periodic_atom_distance_ang": metadata["minimum_periodic_atom_distance_ang"],
        "effective_hBN_strain_percent": metadata["effective_hBN_strain_percent"],
        "n_selected": max(1, limit),
        "siesta_invoked": False,
    }


def build_geometry_only(
    stacking_fdf: Path,
    output_root: Path,
    *,
    approximant: int,
    m: int,
    n: int,
    overwrite: bool,
    min_atom_distance: float = DEFAULT_MIN_ATOM_DISTANCE_ANG,
) -> dict[str, Any]:
    """Materialize the moire geometry + basis + a predict-only manifest, no SIESTA.

    For twist angles whose commensurate cell is too large for a SIESTA
    reference (e.g. ~1.08 deg -> ~11k atoms), the ML models can still predict a
    Hamiltonian from the structure alone. This writes a test_manifest.csv with
    only sample_id + structure_path, which predict_model_on_dataset.py / the
    DeepH adapter consume WITHOUT a reference Hamiltonian. There is no reference,
    so no MAE/relative_frobenius can be computed: the output is an unvalidated
    ML prediction, not a benchmarked one.
    """
    output_root = _safe_output_root(output_root)
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists: {output_root}; pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    _copy_basis(output_root)
    destination = output_root / "splits" / "test" / "0"
    geometry = _prepare_geometry(
        stacking_fdf, destination, approximant=approximant, m=m, n=n, min_atom_distance=min_atom_distance
    )
    _write_test_manifest(
        output_root,
        [
            {
                "sample_id": "moire_0",
                "method": "static_moire_geometry_only",
                "source_run": str(stacking_fdf),
                "source_sample_id": f"{stacking_fdf.parent.name}_0",
                "structure_path": str(destination / "RUN.fdf"),
                "hamiltonian_path": "",  # no SIESTA reference: prediction only
                "run_out_path": "",
                "metadata_path": str(destination / "metadata.json"),
                "valid": True,
                "split": "test",
                "status": "geometry_only",
                "sample_dir": str(destination),
            }
        ],
    )
    _write_json(output_root / "moire_geometry.json", geometry)
    return {
        "dry_run": False,
        "geometry_only": True,
        "siesta_invoked": False,
        "reference_available": False,
        "output_root": str(output_root),
        "test_manifest": str(output_root / "splits" / "test_manifest.csv"),
        "num_atoms": geometry["num_atoms"],
        "twist_angle_deg": geometry["twist_angle_deg"],
        "minimum_periodic_atom_distance_ang": geometry["minimum_periodic_atom_distance_ang"],
        "effective_hBN_strain_percent": geometry["effective_hBN_strain_percent"],
    }


def build_target(
    stacking_fdf: Path,
    output_root: Path,
    *,
    stacking_preset: str,
    approximant: int,
    m: int,
    n: int,
    limit: int,
    siesta_command: str,
    overwrite: bool,
    min_atom_distance: float = DEFAULT_MIN_ATOM_DISTANCE_ANG,
) -> dict[str, Any]:
    output_root = _safe_output_root(output_root)
    if limit < 1:
        raise RuntimeError("--limit must be a positive integer.")
    backup: Path | None = None
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists: {output_root}; pass --overwrite to replace it.")
        backup = output_root.with_name(f".{output_root.name}.backup.{uuid.uuid4().hex}")
        output_root.rename(backup)
    output_root.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    sample_dirs: list[Path] = []
    geometry: dict[str, Any] = {}
    try:
        _copy_basis(output_root)
        for index in range(limit):
            destination = output_root / "splits" / "test" / str(index)
            geometry = _prepare_geometry(
                stacking_fdf,
                destination,
                approximant=approximant,
                m=m,
                n=n,
                min_atom_distance=min_atom_distance,
            )
            stage_required_pseudopotentials(reference_dir=destination, source_dataset_root=COMMON_MATERIAL_ROOT)
            run = run_siesta(destination, command=siesta_command, use_shell=False)
            validation = validate_snapshot(destination)
            if run["returncode"] != 0 or not validation.valid:
                raise RuntimeError(
                    f"SIESTA reference failed for moire sample {index}: returncode={run['returncode']}, "
                    f"errors={validation.errors}, missing={validation.missing_required}"
                )
            sample_dirs.append(destination)
            rows.append(
                {
                    "sample_id": f"moire_{index}",
                    "method": "static_moire",
                    "source_run": str(stacking_fdf),
                    "source_sample_id": f"{stacking_fdf.parent.name}_{index}",
                    "structure_path": str(destination / "RUN.fdf"),
                    "hamiltonian_path": validation.present_artifacts.get("tshs")
                    or validation.present_artifacts["hsx"],
                    "run_out_path": str(destination / "RUN.out"),
                    "metadata_path": str(destination / "metadata.json"),
                    "valid": True,
                    "split": "test",
                    "status": "completed",
                    "sample_dir": str(destination),
                }
            )

        _write_test_manifest(output_root, rows)
        _write_json(
            output_root / "material_provenance.json",
            _material_provenance(
                output_root, stacking_fdf, stacking_preset, geometry, siesta_command, sample_dirs[0]
            ),
        )
        validation = validate_dataset(
            output_root,
            snapshot_dirs=sample_dirs,
            basis_dirs=[output_root / "material_basis"],
            pseudopotential_provenance_paths=[output_root / "material_provenance.json"],
            material_identity_paths=[output_root / "material_provenance.json"],
            siesta_input_paths=[sample_dirs[0] / "RUN.fdf", output_root / "material_provenance.json"],
            validation_profile=G2M_DEEPH_BENCHMARK_PROFILE,
        )
        _write_json(output_root / "artifact_validation.json", validation.to_dict())
        if not validation.valid:
            raise RuntimeError(f"Moire target failed joint artifact validation: {validation.errors}")
        dataset_manifest, frozen = write_benchmark_manifests(
            dataset_root=output_root,
            split_root=output_root / "splits",
            generation_mode="twisted_bilayer_moire_static_siesta",
            strict_paper_ready_provenance=True,
        )
        if backup is not None:
            shutil.rmtree(backup)
        return {
            "dry_run": False,
            "output_root": str(output_root),
            "n_samples": len(rows),
            "num_atoms": geometry["num_atoms"],
            "twist_angle_deg": geometry["twist_angle_deg"],
            "minimum_periodic_atom_distance_ang": geometry["minimum_periodic_atom_distance_ang"],
            "effective_hBN_strain_percent": geometry["effective_hBN_strain_percent"],
            "split_counts": frozen["split_counts"],
            "benchmark_dataset_id": dataset_manifest["benchmark_dataset_id"],
            "benchmark_ready": dataset_manifest["benchmark_ready"],
        }
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        if backup is not None and backup.exists():
            backup.rename(output_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacking-preset", default=DEFAULT_STACKING)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--approximant", type=int, default=2, help="Legacy compatibility option; must be 2.")
    parser.add_argument("--commensurate-angle", default="1,2", help="coprime (m,n) e.g. '1,2'.")
    parser.add_argument("--min-atom-distance", type=float, default=DEFAULT_MIN_ATOM_DISTANCE_ANG)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--siesta-command", default="siesta")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Materialize geometry + predict-only manifest without SIESTA (for cells too large to reference, e.g. ~1.08 deg). Output is an unvalidated ML prediction target.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    m, n = (int(x) for x in str(args.commensurate_angle).split(","))
    stacking_fdf = REPO_ROOT / "materials" / args.stacking_preset / "RUN.fdf"
    if not stacking_fdf.is_file():
        raise RuntimeError(f"Stacking preset RUN.fdf not found: {stacking_fdf}")

    if args.output_root is not None:
        output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    else:
        angle = commensurate_angle_degrees(m, n)
        output_root = REPO_ROOT / "Comparison" / "datasets" / f"graphene_hBN_moire_{angle:.0f}deg"

    if args.dry_run:
        result = dry_run_plan(
            stacking_fdf,
            approximant=args.approximant,
            m=m,
            n=n,
            limit=args.limit,
            min_atom_distance=args.min_atom_distance,
        )
    elif args.geometry_only:
        result = build_geometry_only(
            stacking_fdf,
            output_root,
            approximant=args.approximant,
            m=m,
            n=n,
            overwrite=args.overwrite,
            min_atom_distance=args.min_atom_distance,
        )
    else:
        result = build_target(
            stacking_fdf,
            output_root,
            stacking_preset=args.stacking_preset,
            approximant=args.approximant,
            m=m,
            n=n,
            limit=args.limit,
            siesta_command=args.siesta_command,
            overwrite=args.overwrite,
            min_atom_distance=args.min_atom_distance,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
