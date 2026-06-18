#!/usr/bin/env python3
"""Build finite-displacement Hamiltonian derivative stencil structures."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fdf_materialization import extract_fdf_structure, materialize_sample_fdf  # noqa: E402


AXES = {"x": 0, "y": 1, "z": 2}
VALID_METHODS = {"central", "forward", "backward"}
STRUCTURE_COPY_SUFFIXES = {".psf", ".psml", ".vps", ".ion", ".xml"}
MATRIX_SUFFIXES = {".HSX", ".TSHS", ".TSDE", ".nc"}
OUTPUT_SUFFIXES = {".out", ".XV", ".STRUCT_OUT", ".ORB_INDX"}


class DerivativeStencilBuildError(RuntimeError):
    """Raised when derivative stencil structures cannot be built safely."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DerivativeStencilBuildError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DerivativeStencilBuildError(f"Malformed JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DerivativeStencilBuildError(f"JSON payload must be an object: {path}")
    return payload


def parse_csv_list(value: str | None, *, field: str) -> list[str]:
    if value in (None, ""):
        return []
    items = [item.strip() for item in str(value).split(",")]
    parsed = [item for item in items if item]
    if not parsed:
        raise DerivativeStencilBuildError(f"{field} cannot be empty.")
    return parsed


def parse_delta_values(values: list[str]) -> list[float]:
    deltas: list[float] = []
    for value in values:
        for item in parse_csv_list(value, field="delta_ang"):
            try:
                delta = float(item)
            except ValueError as exc:
                raise DerivativeStencilBuildError(f"delta_ang must be numeric in Ang: {item!r}") from exc
            if delta <= 0:
                raise DerivativeStencilBuildError("delta_ang values must be positive.")
            if delta not in deltas:
                deltas.append(delta)
    if not deltas:
        raise DerivativeStencilBuildError("At least one --delta-ang value is required.")
    return deltas


def parse_axes(value: str) -> list[str]:
    axes = [axis.lower() for axis in parse_csv_list(value, field="axes")]
    invalid = [axis for axis in axes if axis not in AXES]
    if invalid:
        raise DerivativeStencilBuildError(f"Unsupported axes: {', '.join(invalid)}. Use x, y, z.")
    return axes


def parse_atom_indices(value: str) -> list[int]:
    atoms: list[int] = []
    for token in parse_csv_list(value, field="atoms"):
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise DerivativeStencilBuildError(f"Invalid atom range: {token!r}") from exc
            if start > end:
                raise DerivativeStencilBuildError(f"Invalid descending atom range: {token!r}")
            candidates = range(start, end + 1)
        else:
            try:
                candidates = [int(token)]
            except ValueError as exc:
                raise DerivativeStencilBuildError(f"Invalid atom index: {token!r}") from exc
        for atom in candidates:
            if atom < 0:
                raise DerivativeStencilBuildError("Atom indices are zero-based and must be non-negative.")
            if atom not in atoms:
                atoms.append(atom)
    return atoms


def safe_sample_id(*parts: Any) -> str:
    raw = "_".join(str(part) for part in parts if str(part) != "")
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw)
    safe = safe.strip("._")
    if not safe or safe in {".", ".."}:
        raise DerivativeStencilBuildError(f"Unsafe generated sample id from parts: {parts!r}")
    return safe


def load_base_rows(
    *,
    source_dataset_root: Path,
    frozen_split: Path | None,
    split: str,
) -> list[dict[str, Any]]:
    if frozen_split is None:
        candidate = source_dataset_root / "frozen_split_manifest.json"
        frozen_split = candidate if candidate.exists() else None
    if frozen_split is not None:
        payload = read_json(frozen_split)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise DerivativeStencilBuildError(f"frozen split manifest has no rows list: {frozen_split}")
        selected = [dict(row) for row in rows if split == "all" or str(row.get("split") or "") == split]
        if not selected:
            raise DerivativeStencilBuildError(f"No frozen split rows selected for split {split!r}.")
        return selected

    split_root = source_dataset_root / "splits" / split
    if not split_root.exists():
        raise DerivativeStencilBuildError(
            f"Provide --frozen-split or a source dataset with splits/{split}: {source_dataset_root}"
        )
    rows = [
        {"sample_id": path.name, "split": split, "sample_dir": str(path)}
        for path in sorted(split_root.iterdir())
        if path.is_dir() and (path / "RUN.fdf").exists()
    ]
    if not rows:
        raise DerivativeStencilBuildError(f"No sample directories with RUN.fdf under {split_root}")
    return rows


def load_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    path = sample_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DerivativeStencilBuildError(f"Malformed sample metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DerivativeStencilBuildError(f"metadata.json must be an object: {path}")
    return payload


def base_sample_id(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id") or row.get("graph2mat_sample_id") or row.get("deeph_sample_id") or "").strip()
    if sample_id:
        return sample_id
    sample_dir = str(row.get("sample_dir") or "").strip()
    if sample_dir:
        return Path(sample_dir).name
    raise DerivativeStencilBuildError(f"Frozen split row is missing sample_id and sample_dir: {row}")


def selected_base_rows(
    rows: list[dict[str, Any]],
    *,
    sample_ids: list[str],
    max_base_snapshots: int | None,
) -> list[dict[str, Any]]:
    if sample_ids:
        allowed = set(sample_ids)
        rows = [row for row in rows if base_sample_id(row) in allowed]
        missing = sorted(allowed - {base_sample_id(row) for row in rows})
        if missing:
            raise DerivativeStencilBuildError(f"Selected base snapshots not found: {', '.join(missing)}")
    if max_base_snapshots is not None:
        if max_base_snapshots <= 0:
            raise DerivativeStencilBuildError("--max-base-snapshots must be positive.")
        rows = rows[:max_base_snapshots]
    if not rows:
        raise DerivativeStencilBuildError("No base snapshots selected.")
    return rows


def copy_support_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.name in {"RUN.fdf", "metadata.json"}:
            continue
        if source.suffix in MATRIX_SUFFIXES or source.suffix in OUTPUT_SUFFIXES:
            continue
        if source.suffix in STRUCTURE_COPY_SUFFIXES:
            shutil.copy2(source, target_dir / source.name)
            copied.append(source.name)
    return copied


def inherited_hashes(metadata: dict[str, Any]) -> dict[str, str]:
    fields = (
        "material_compatibility_hash",
        "orbital_ordering_hash",
        "neighbor_list_hash",
        "sparsity_pattern_hash",
        "basis_hash",
        "pseudopotential_hash",
    )
    return {field: str(metadata[field]) for field in fields if str(metadata.get(field) or "").strip()}


def missing_hash_fields(metadata: dict[str, Any]) -> list[str]:
    fields = (
        "material_compatibility_hash",
        "orbital_ordering_hash",
        "neighbor_list_hash",
        "sparsity_pattern_hash",
        "basis_hash",
        "pseudopotential_hash",
    )
    return [field for field in fields if not str(metadata.get(field) or "").strip()]


def displaced_positions(
    positions: list[tuple[float, float, float]],
    *,
    atom_index_zero_based: int,
    axis_index: int,
    signed_delta: float,
) -> list[list[float]]:
    if atom_index_zero_based < 0 or atom_index_zero_based >= len(positions):
        raise DerivativeStencilBuildError(
            f"atom index {atom_index_zero_based} is outside base structure with {len(positions)} atoms."
        )
    updated = [list(position) for position in positions]
    updated[atom_index_zero_based][axis_index] += signed_delta
    return updated


def write_structure_sample(
    *,
    base_run_fdf: Path,
    base_structure: Any,
    source_sample_dir: Path,
    output_sample_dir: Path,
    sample_id: str,
    positions_ang: list[list[float]] | list[tuple[float, float, float]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_sample_dir.mkdir(parents=True, exist_ok=False)
    copied_support_files = copy_support_files(source_sample_dir, output_sample_dir)
    materialized = materialize_sample_fdf(
        base_run_fdf,
        output_sample_dir / "RUN.fdf",
        positions_ang=positions_ang,
        atom_species=base_structure.atom_species,
        lattice_vectors_ang=base_structure.lattice_vectors_ang,
        system_label=sample_id,
        system_name=sample_id,
        structure_type=base_structure.structure_type,
    )
    payload = {**materialized.metadata, **metadata}
    payload["support_files_copied"] = copied_support_files
    write_json(output_sample_dir / "metadata.json", payload)
    return {
        "sample_id": sample_id,
        "sample_dir": str(output_sample_dir),
        "run_fdf": str(output_sample_dir / "RUN.fdf"),
        "metadata_path": str(output_sample_dir / "metadata.json"),
        "sign": payload.get("sign"),
        "sign_label": payload.get("sign_label"),
        "atom_index_zero_based": payload.get("atom_index_zero_based"),
        "axis": payload.get("axis"),
        "delta_ang": payload.get("delta_ang"),
    }


def signs_for_method(method: str) -> list[int]:
    if method == "central":
        return [1, -1]
    if method == "forward":
        return [1]
    if method == "backward":
        return [-1]
    raise DerivativeStencilBuildError(f"Unsupported finite difference method: {method!r}")


def build_derivative_stencils(
    *,
    source_dataset_root: Path,
    output_stencil_root: Path,
    frozen_split: Path | None = None,
    split: str = "test",
    method: str = "central",
    delta_ang_values: list[float],
    atom_indices_zero_based: list[int],
    axes: list[str],
    base_sample_ids: list[str] | None = None,
    max_base_snapshots: int | None = None,
    include_base: bool | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    method = str(method or "central").strip().lower()
    if method not in VALID_METHODS:
        raise DerivativeStencilBuildError(f"--method must be one of: {', '.join(sorted(VALID_METHODS))}.")
    if not atom_indices_zero_based:
        raise DerivativeStencilBuildError("At least one zero-based atom index is required.")
    if not axes:
        raise DerivativeStencilBuildError("At least one axis is required.")
    include_base = True if include_base is None else bool(include_base)
    if output_stencil_root.exists() and any(output_stencil_root.iterdir()):
        if not overwrite:
            raise DerivativeStencilBuildError(
                f"Output stencil root already exists and is not empty: {output_stencil_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_stencil_root)
    structures_root = output_stencil_root / "structures"
    structures_root.mkdir(parents=True, exist_ok=True)

    rows = load_base_rows(source_dataset_root=source_dataset_root, frozen_split=frozen_split, split=split)
    rows = selected_base_rows(
        rows,
        sample_ids=base_sample_ids or [],
        max_base_snapshots=max_base_snapshots,
    )

    sample_records: list[dict[str, Any]] = []
    stencil_records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for row in rows:
        base_id = base_sample_id(row)
        source_sample_dir = Path(str(row.get("sample_dir") or ""))
        if not source_sample_dir.exists():
            raise DerivativeStencilBuildError(f"Base sample_dir does not exist for {base_id}: {source_sample_dir}")
        base_run_fdf = source_sample_dir / "RUN.fdf"
        if not base_run_fdf.exists():
            raise DerivativeStencilBuildError(f"Base sample is missing RUN.fdf: {base_run_fdf}")
        base_metadata = load_sample_metadata(source_sample_dir)
        base_structure = extract_fdf_structure(base_run_fdf, structure_type=base_metadata.get("structure_type"))
        hashes = inherited_hashes(base_metadata)
        missing_hashes = missing_hash_fields(base_metadata)
        if missing_hashes:
            warnings.append(
                {
                    "kind": "missing_diagnostic_hashes",
                    "base_sample_id": base_id,
                    "missing_fields": missing_hashes,
                    "message": "Compatibility hashes were not present in base metadata and were not fabricated.",
                }
            )
        material_label = str(
            base_metadata.get("material_label")
            or base_metadata.get("material_id")
            or base_metadata.get("base_material_label")
            or source_dataset_root.name
        )
        base_system_label = str(
            base_metadata.get("base_system_label")
            or base_metadata.get("reference_system_label")
            or base_metadata.get("system_group_label")
            or base_id
        )
        if include_base:
            base_sample = safe_sample_id(base_id, "base")
            metadata = {
                "id": base_sample,
                "sample_id": base_sample,
                "base_sample_id": base_id,
                "source_base_sample_id": base_id,
                "source_sample_dir": str(source_sample_dir),
                "generation_mode": "hamiltonian_derivative_stencil",
                "finite_difference_method": method,
                "material_label": material_label,
                "base_system_label": base_system_label,
                "is_reference": True,
                "sign": 0,
                "sign_label": "0",
                "amplitude_ang": 0.0,
                "delta_ang": 0.0,
                "displacement_ang": [0.0, 0.0, 0.0],
                "split": split,
                "claim_status": "diagnostic_only",
                "hamiltonian_units": "eV",
                "displacement_units": "Ang",
                "derivative_units": "eV/Ang",
                "diagnostic_missing_hashes": missing_hashes,
                **hashes,
            }
            sample_records.append(
                write_structure_sample(
                    base_run_fdf=base_run_fdf,
                    base_structure=base_structure,
                    source_sample_dir=source_sample_dir,
                    output_sample_dir=structures_root / base_sample,
                    sample_id=base_sample,
                    positions_ang=base_structure.positions_ang,
                    metadata=metadata,
                )
            )

        for delta_ang in delta_ang_values:
            for atom_index in atom_indices_zero_based:
                for axis in axes:
                    axis_index = AXES[axis]
                    split_group_id = safe_sample_id("dH", base_id, f"atom{atom_index:04d}", axis, f"delta{delta_ang:g}")
                    generated_samples: dict[str, str] = {}
                    for sign in signs_for_method(method):
                        sign_label = "plus" if sign > 0 else "minus"
                        sample_id = safe_sample_id(base_id, f"atom{atom_index:04d}", axis, f"d{delta_ang:g}", sign_label)
                        displacement = [0.0, 0.0, 0.0]
                        displacement[axis_index] = sign * delta_ang
                        positions = displaced_positions(
                            base_structure.positions_ang,
                            atom_index_zero_based=atom_index,
                            axis_index=axis_index,
                            signed_delta=sign * delta_ang,
                        )
                        metadata = {
                            "id": sample_id,
                            "sample_id": sample_id,
                            "base_sample_id": base_id,
                            "source_base_sample_id": base_id,
                            "source_sample_dir": str(source_sample_dir),
                            "generation_mode": "hamiltonian_derivative_stencil",
                            "finite_difference_method": method,
                            "material_label": material_label,
                            "base_system_label": base_system_label,
                            "is_reference": False,
                            "atom_index": atom_index + 1,
                            "atom_index_zero_based": atom_index,
                            "axis": axis,
                            "axis_index": axis_index,
                            "sign": sign,
                            "sign_label": "+" if sign > 0 else "-",
                            "amplitude_ang": delta_ang,
                            "delta_ang": delta_ang,
                            "displacement_ang": displacement,
                            "split": split,
                            "split_group_id": split_group_id,
                            "claim_status": "diagnostic_only",
                            "hamiltonian_units": "eV",
                            "displacement_units": "Ang",
                            "derivative_units": "eV/Ang",
                            "diagnostic_missing_hashes": missing_hashes,
                            **hashes,
                        }
                        sample_records.append(
                            write_structure_sample(
                                base_run_fdf=base_run_fdf,
                                base_structure=base_structure,
                                source_sample_dir=source_sample_dir,
                                output_sample_dir=structures_root / sample_id,
                                sample_id=sample_id,
                                positions_ang=positions,
                                metadata=metadata,
                            )
                        )
                        generated_samples["plus_sample_id" if sign > 0 else "minus_sample_id"] = sample_id
                    stencil_records.append(
                        {
                            "base_sample_id": base_id,
                            "atom_index_zero_based": atom_index,
                            "axis": axis,
                            "axis_index": axis_index,
                            "delta_ang": delta_ang,
                            "finite_difference_method": method,
                            "split_group_id": split_group_id,
                            **generated_samples,
                        }
                    )

    manifest = {
        "schema": "hamiltonian_derivative_stencil_structures_v1",
        "source_dataset_root": str(source_dataset_root),
        "frozen_split": str(frozen_split) if frozen_split else "",
        "output_stencil_root": str(output_stencil_root),
        "structures_root": str(structures_root),
        "finite_difference_method": method,
        "preferred_benchmark_method": "central stencil with R+ and R- and central finite difference",
        "delta_ang_values": delta_ang_values,
        "split": split,
        "base_snapshots": [base_sample_id(row) for row in rows],
        "atom_indices_zero_based": atom_indices_zero_based,
        "axes": axes,
        "include_base": include_base,
        "samples": sample_records,
        "stencils": stencil_records,
        "sample_count": len(sample_records),
        "stencil_count": len(stencil_records),
        "warnings": warnings,
        "matrix_outputs_created": False,
        "siesta_run": False,
        "ml_predictions_run": False,
        "derivative_metrics_run": False,
    }
    write_json(output_stencil_root / "derivative_stencil_manifest.json", manifest)
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--frozen-split", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-stencil-root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(VALID_METHODS), default="central")
    parser.add_argument("--delta-ang", nargs="+", required=True, help="One or more positive Ang values; comma-separated values are accepted.")
    parser.add_argument("--base-sample-id", action="append", default=[])
    parser.add_argument("--max-base-snapshots", type=int, default=None)
    parser.add_argument("--atoms", required=True, help="Zero-based atom indices, e.g. 0,2 or 0-3.")
    parser.add_argument("--axes", required=True, help="Comma-separated Cartesian axes: x,y,z.")
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    manifest = build_derivative_stencils(
        source_dataset_root=args.source_dataset_root,
        frozen_split=args.frozen_split,
        split=str(args.split or "test"),
        output_stencil_root=args.output_stencil_root,
        method=args.method,
        delta_ang_values=parse_delta_values(args.delta_ang),
        atom_indices_zero_based=parse_atom_indices(args.atoms),
        axes=parse_axes(args.axes),
        base_sample_ids=args.base_sample_id,
        max_base_snapshots=args.max_base_snapshots,
        include_base=args.include_base,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output_stencil_root": manifest["output_stencil_root"], "sample_count": manifest["sample_count"], "stencil_count": manifest["stencil_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
