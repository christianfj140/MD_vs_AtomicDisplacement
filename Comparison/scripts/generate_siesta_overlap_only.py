#!/usr/bin/env python3
"""Generate the exact SIESTA PAO overlap without SCF and export DeepH blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse.linalg


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fdf_materialization import (  # noqa: E402
    _replace_or_append_block,
    _set_fdf_directive,
    extract_fdf_structure,
)
from material_presets import resolve_material_bundle  # noqa: E402


KPOINTS = {
    "Gamma": (0.0, 0.0, 0.0),
    "K": (1.0 / 3.0, 1.0 / 3.0, 0.0),
    "M": (0.5, 0.0, 0.0),
}


def open_block_h5(path: Path) -> h5py.File:
    """Create DeepH's many-small-dataset format without metadata-cache thrashing."""
    access = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    cache = access.get_mdc_config()
    # ponytail: keep metadata in RAM; revisit only if block files exceed host memory.
    cache.evictions_enabled = 0
    cache.incr_mode = 0
    cache.flash_incr_mode = 0
    cache.decr_mode = 0
    access.set_mdc_config(cache)
    file_id = h5py.h5f.create(os.fsencode(path), fapl=access)
    return h5py.File(file_id)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def overlap_only_fdf(source: Path, *, kgrid: int) -> str:
    text = source.read_text(encoding="utf-8")
    text = _set_fdf_directive(text, "MaxSCFIterations", "0")
    text = _set_fdf_directive(text, "TS.onlyS", "T")
    text = _set_fdf_directive(text, "TS.HS.Save", "T")
    text = _replace_or_append_block(
        text,
        "kgrid_Monkhorst_Pack",
        [f"  {kgrid}  0  0  0.0", f"  0  {kgrid}  0  0.0", "  0  0  1  0.0"],
    )
    return text


def parse_time_v(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    aliases = {
        "Elapsed (wall clock) time (h:mm:ss or m:ss)": "wall_clock",
        "Maximum resident set size (kbytes)": "max_rss_kib",
        "User time (seconds)": "user_seconds",
        "System time (seconds)": "system_seconds",
        "File system outputs": "filesystem_outputs",
    }
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for label, key in aliases.items():
            marker = f"{label}: "
            if line.strip().startswith(marker):
                raw = line.strip()[len(marker) :]
                try:
                    values[key] = float(raw) if "." in raw else int(raw)
                except ValueError:
                    values[key] = raw
    if "max_rss_kib" in values:
        values["max_rss_gib"] = float(values["max_rss_kib"]) / 1024**2
    return values


def _orbital_rows(path: Path) -> list[dict[str, int]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0].split()
    try:
        primary_count = int(header[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"{path}: invalid ORB_INDX header") from exc
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 13 or not parts[0].isdigit():
            continue
        try:
            io = int(parts[0]) - 1
            if io >= primary_count:
                break
            rows.append(
                {
                    "io": io,
                    "atom": int(parts[1]) - 1,
                    "n": int(parts[5]),
                    "l": int(parts[6]),
                    "m": int(parts[7]),
                    "zeta": int(parts[8]),
                }
            )
        except ValueError:
            continue
    if len(rows) != primary_count or [row["io"] for row in rows] != list(range(primary_count)):
        raise RuntimeError(f"{path}: cannot identify contiguous primary-cell orbitals")
    return rows


def _deeph_orbital_map(rows: list[dict[str, int]]) -> tuple[list[int], list[int], list[list[int]]]:
    atom_count = max(row["atom"] for row in rows) + 1
    mapping = [0] * len(rows)
    signs = [1] * len(rows)
    orbital_types: list[list[int]] = []
    m_order = {
        0: {0: 0},
        1: {-1: 0, 0: 1, 1: -1},
        2: {-2: 0, -1: 2, 0: -2, 1: 1, 2: -1},
        3: {-3: 0, -2: 1, -1: -1, 0: 2, 1: -2, 2: 3, 3: -3},
    }
    for atom in range(atom_count):
        atom_rows = [row for row in rows if row["atom"] == atom]
        ranked = sorted(
            enumerate(atom_rows),
            key=lambda item: (
                1000 * item[1]["l"]
                + 100 * item[1]["n"]
                + 10 * item[1]["zeta"]
                + m_order[item[1]["l"]][item[1]["m"]]
            ),
        )
        for deep_index, (old_local_index, row) in enumerate(ranked):
            mapping[row["io"]] = deep_index
            signs[row["io"]] = -1 if row["m"] % 2 else 1
        shells: list[int] = []
        for ell in sorted({row["l"] for row in atom_rows}):
            count = sum(row["l"] == ell for row in atom_rows)
            degeneracy = 2 * ell + 1
            if count % degeneracy:
                raise RuntimeError(f"Incomplete l={ell} shell for atom {atom + 1}")
            shells.extend([ell] * (count // degeneracy))
        orbital_types.append(shells)
    return mapping, signs, orbital_types


def _deeph_blocks(matrix: Any, rows: list[dict[str, int]]) -> tuple[dict, list[list[int]], float]:
    mapping, signs, orbital_types = _deeph_orbital_map(rows)
    no = len(rows)
    if matrix.no != no:
        raise RuntimeError(f"Matrix has {matrix.no} orbitals but ORB_INDX has {no}")
    offsets = np.cumsum([0, *[sum(2 * ell + 1 for ell in shells) for shells in orbital_types]])
    blocks: dict[tuple[int, int, int, int, int], np.ndarray] = {}
    csr = matrix._csr
    for row_orbital in range(no):
        start = int(csr.ptr[row_orbital])
        stop = start + int(csr.ncol[row_orbital])
        atom_i = rows[row_orbital]["atom"]
        for cursor in range(start, stop):
            sc_index, col_orbital = divmod(int(csr.col[cursor]), no)
            atom_j = rows[col_orbital]["atom"]
            r = tuple(int(value) for value in matrix.lattice.sc_off[sc_index])
            key = (*r, atom_i + 1, atom_j + 1)
            block = blocks.setdefault(
                key,
                np.zeros(
                    (int(offsets[atom_i + 1] - offsets[atom_i]), int(offsets[atom_j + 1] - offsets[atom_j])),
                    dtype=np.float64,
                ),
            )
            block[mapping[row_orbital], mapping[col_orbital]] += (
                float(csr._D[cursor, 0]) * signs[row_orbital] * signs[col_orbital]
            )
    raw_norm_sq = sum(float(np.vdot(block, block).real) for block in blocks.values())
    adjustment_sq = 0.0
    visited: set[tuple[int, int, int, int, int]] = set()
    for key in list(blocks):
        if key in visited:
            continue
        inverse = (-key[0], -key[1], -key[2], key[4], key[3])
        left = blocks[key]
        right = blocks.get(inverse, np.zeros((left.shape[1], left.shape[0]), dtype=left.dtype))
        averaged = 0.5 * (left + right.T)
        adjustment_sq += float(np.vdot(averaged - left, averaged - left).real)
        adjustment_sq += float(np.vdot(averaged.T - right, averaged.T - right).real)
        blocks[key] = averaged
        blocks[inverse] = averaged.T
        visited.update((key, inverse))
    return blocks, orbital_types, math.sqrt(adjustment_sq / max(raw_norm_sq, 1e-300))


def export_deeph_overlap(
    onlys_path: Path,
    orb_indx_path: Path,
    run_fdf: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import sisl

    overlap = sisl.get_sile(str(onlys_path)).read_overlap()
    rows = _orbital_rows(orb_indx_path)
    blocks, orbital_types, adjustment = _deeph_blocks(overlap, rows)
    no = len(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "overlaps.h5"
    with open_block_h5(h5_path) as handle:
        for key, block in sorted(blocks.items()):
            handle.create_dataset(str(list(key)), data=block)

    structure = extract_fdf_structure(run_fdf, structure_type="crystal")
    lattice = np.asarray(structure.lattice_vectors_ang, dtype=float)
    positions = np.asarray(structure.positions_ang, dtype=float)
    label_by_index = {species.index: species.atomic_number for species in structure.species}
    elements = np.asarray([label_by_index[index] for index in structure.atom_species], dtype=int)
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    np.savetxt(output_dir / "lat.dat", lattice.T, fmt="%.18e")
    np.savetxt(output_dir / "rlat.dat", reciprocal.T, fmt="%.18e")
    np.savetxt(output_dir / "site_positions.dat", positions.T, fmt="%.18e")
    np.savetxt(output_dir / "element.dat", elements, fmt="%d")
    (output_dir / "orbital_types.dat").write_text(
        "".join("  ".join(str(ell) for ell in shells) + "\n" for shells in orbital_types),
        encoding="utf-8",
    )
    (output_dir / "R_list.dat").write_text(
        "".join(f"{r[0]} {r[1]} {r[2]}\n" for r in sorted({key[:3] for key in blocks})),
        encoding="utf-8",
    )
    write_json(
        output_dir / "info.json",
        {"nsites": structure.atom_count, "isorthogonal": False, "isspinful": False, "norbits": no},
    )
    return {
        "overlaps_h5": str(h5_path),
        "overlaps_h5_sha256": file_sha256(h5_path),
        "n_atoms": structure.atom_count,
        "n_orbitals": no,
        "n_blocks": len(blocks),
        "n_nonzero": int(overlap.nnz),
        "nsc": overlap.lattice.nsc.tolist(),
        "r_vectors": len({key[:3] for key in blocks}),
        "basis_transform": "siesta_real_orbitals_to_deeph_m_zeta_l_order_with_phase",
        "canonical_hermitization": {
            "policy": "pair_average_S_R_with_transpose_S_minus_R",
            "relative_frobenius_adjustment": adjustment,
        },
    }


def _assemble_h5(path: Path, orbital_types_path: Path, kpoint: tuple[float, float, float]) -> np.ndarray:
    counts = [
        sum(2 * int(value) + 1 for value in line.split())
        for line in orbital_types_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    offsets = np.cumsum([0, *counts])
    matrix = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=np.complex128)
    with h5py.File(path, "r") as handle:
        for raw_key in handle:
            rx, ry, rz, atom_i, atom_j = json.loads(raw_key)
            block = np.asarray(handle[raw_key])
            phase = np.exp(2j * np.pi * np.dot(kpoint, (rx, ry, rz)))
            matrix[offsets[atom_i - 1] : offsets[atom_i], offsets[atom_j - 1] : offsets[atom_j]] += block * phase
    return matrix


def _sparse_extreme_eigenvalues(matrix: scipy.sparse.spmatrix) -> tuple[float, float]:
    options = {
        "k": 1,
        "return_eigenvectors": False,
        "tol": 1e-6,
        "maxiter": 5000,
        "ncv": min(80, matrix.shape[0] - 1),
    }
    minimum = scipy.sparse.linalg.eigsh(matrix, which="SA", **options)[0]
    maximum = scipy.sparse.linalg.eigsh(matrix, which="LA", **options)[0]
    return float(minimum), float(maximum)


def overlap_diagnostics(
    onlys_path: Path,
    overlaps_h5: Path,
    orbital_types_path: Path,
    *,
    dense_limit: int = 5000,
) -> dict[str, Any]:
    import sisl

    overlap = sisl.get_sile(str(onlys_path)).read_overlap()
    rows = []
    passed = True
    for label, kpoint in KPOINTS.items():
        raw_matrix = overlap.Sk(kpoint, format="csr")
        raw_norm = scipy.sparse.linalg.norm(raw_matrix)
        raw_hermiticity = float(
            scipy.sparse.linalg.norm(raw_matrix - raw_matrix.getH()) / max(raw_norm, 1e-30)
        )
        row: dict[str, Any] = {
            "label": label,
            "k": kpoint,
            "raw_onlys_hermiticity_relative_frobenius": raw_hermiticity,
        }
        if raw_matrix.shape[0] <= dense_limit:
            matrix = _assemble_h5(overlaps_h5, orbital_types_path, kpoint)
            norm = np.linalg.norm(matrix)
            hermiticity = float(np.linalg.norm(matrix - matrix.conj().T) / max(norm, 1e-30))
            row["hermiticity_relative_frobenius"] = hermiticity
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
            row.update(
                {
                    "minimum_eigenvalue": float(eigenvalues[0]),
                    "maximum_eigenvalue": float(eigenvalues[-1]),
                    "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                    "positivity_status": "validated_dense",
                }
            )
            passed = passed and hermiticity < 1e-10 and eigenvalues[0] > 0
        else:
            started = time.time()
            matrix = 0.5 * (raw_matrix + raw_matrix.getH())
            row["hermiticity_relative_frobenius"] = 0.0
            row["hermiticity_status"] = "guaranteed_by_canonical_R_pair_hermitization"
            try:
                minimum, maximum = _sparse_extreme_eigenvalues(matrix)
            except scipy.sparse.linalg.ArpackNoConvergence as exc:
                row.update(
                    {
                        "positivity_status": "arpack_not_converged",
                        "positivity_error": str(exc),
                        "positivity_elapsed_seconds": time.time() - started,
                    }
                )
                passed = False
            else:
                row.update(
                    {
                        "minimum_eigenvalue": minimum,
                        "maximum_eigenvalue": maximum,
                        "condition_number": maximum / minimum,
                        "positivity_status": "validated_sparse_arpack",
                        "positivity_matrix_source": "canonical_equivalent_symmetrized_raw_onlyS",
                        "positivity_elapsed_seconds": time.time() - started,
                    }
                )
                passed = passed and minimum > 0
        rows.append(row)
    return {
        "status": "valid" if passed else "invalid",
        "no_identity_overlap": True,
        "source": "SIESTA_TS.onlyS_exact_PAO_integrals",
        "kpoints": rows,
    }


def stage_pseudopotentials(output_dir: Path, preset: str) -> dict[str, str]:
    validated = resolve_material_bundle(
        {"material": {"preset": preset}},
        allow_legacy_default=False,
    ).validated
    hashes = {}
    for label, source in validated.pseudopotentials.items():
        destination = output_dir / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(os.path.relpath(source, output_dir), destination)
        hashes[label] = file_sha256(source)
    return hashes


def generate(
    run_fdf: Path,
    output_dir: Path,
    *,
    preset: str,
    siesta_command: str,
    kgrid: int,
    overwrite: bool,
) -> dict[str, Any]:
    if kgrid < 3 or kgrid % 2 == 0:
        raise RuntimeError("--kgrid must be an odd integer >= 3 so periodic R blocks are retained")
    effective_text = overlap_only_fdf(run_fdf, kgrid=kgrid)
    reusable_raw = (
        output_dir.is_dir()
        and (output_dir / "RUN.fdf").is_file()
        and (output_dir / "RUN.fdf").read_text(encoding="utf-8") == effective_text
        and len(list(output_dir.glob("*.onlyS"))) == 1
        and len(list(output_dir.glob("*.ORB_INDX"))) == 1
        and (output_dir / "time-v.txt").is_file()
    )
    if output_dir.exists():
        if not overwrite and not reusable_raw:
            raise RuntimeError(f"Output exists: {output_dir}; pass --overwrite")
        if not reusable_raw:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_fdf = output_dir / "RUN.fdf"
    effective_fdf.write_text(effective_text, encoding="utf-8")
    pseudo_hashes = stage_pseudopotentials(output_dir, preset)
    stdout_path = output_dir / "siesta.stdout.log"
    stderr_path = output_dir / "siesta.stderr.log"
    time_path = output_dir / "time-v.txt"
    command = ["/usr/bin/time", "-v", "-o", str(time_path), siesta_command]
    started = time.time()
    returncode = 0
    if not reusable_raw:
        with effective_fdf.open("rb") as stdin, stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(command, cwd=output_dir, stdin=stdin, stdout=stdout, stderr=stderr, check=False)
        returncode = completed.returncode
    onlys_files = sorted(output_dir.glob("*.onlyS"))
    if returncode != 0 or len(onlys_files) != 1:
        raise RuntimeError(
            f"SIESTA overlap-only failed: returncode={returncode}, onlyS_files={onlys_files}; "
            f"see {stdout_path} and {stderr_path}"
        )
    onlys_path = onlys_files[0]
    orb_indx_files = sorted(output_dir.glob("*.ORB_INDX"))
    if len(orb_indx_files) != 1:
        raise RuntimeError(f"Expected one ORB_INDX, found {orb_indx_files}")
    exported = export_deeph_overlap(onlys_path, orb_indx_files[0], effective_fdf, output_dir)
    diagnostics = overlap_diagnostics(
        onlys_path,
        Path(exported["overlaps_h5"]),
        output_dir / "orbital_types.dat",
    )
    write_json(output_dir / "diagnostics.json", diagnostics)
    manifest = {
        "status": "completed" if diagnostics["status"] == "valid" else "invalid",
        "campaign_contract": "geometry_plus_exact_overlap_no_reference_hamiltonian",
        "reference_hamiltonian_generated": False,
        "scf_iterations": 0,
        "siesta_onlys": True,
        "identity_overlap_used": False,
        "source_run_fdf": str(run_fdf.resolve()),
        "effective_run_fdf": str(effective_fdf),
        "source_run_fdf_sha256": file_sha256(run_fdf),
        "effective_run_fdf_sha256": file_sha256(effective_fdf),
        "raw_onlys": str(onlys_path),
        "raw_onlys_sha256": file_sha256(onlys_path),
        "preset": preset,
        "pseudopotential_sha256": pseudo_hashes,
        "kgrid": [kgrid, kgrid, 1],
        "command": command,
        "returncode": returncode,
        "siesta_raw_reused": reusable_raw,
        "elapsed_seconds_python": time.time() - started,
        "resources": parse_time_v(time_path),
        "export": exported,
        "diagnostics": diagnostics,
    }
    write_json(output_dir / "overlap_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-fdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", default="bilayer_graphene_hBN_AA")
    parser.add_argument("--siesta-command", default="/home/christian/bin/siesta")
    parser.add_argument("--kgrid", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = generate(
        args.run_fdf.resolve(),
        args.output_dir.resolve(),
        preset=args.preset,
        siesta_command=args.siesta_command,
        kgrid=args.kgrid,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
