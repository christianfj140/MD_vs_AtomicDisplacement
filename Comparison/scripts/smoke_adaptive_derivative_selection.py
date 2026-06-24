#!/usr/bin/env python3
"""Smoke-test adaptive derivative base selection without SIESTA or training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
for directory in (SCRIPT_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_hamiltonian_derivative_stencils import (  # noqa: E402
    build_derivative_stencils,
    parse_atom_indices,
    parse_axes,
)


DEFAULT_PAYLOAD = REPO_ROOT / "Comparison" / "config" / "adaptive_derivative_selection_smoke.json"


class AdaptiveSelectionSmokeError(RuntimeError):
    """Raised when the adaptive selection smoke fails."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdaptiveSelectionSmokeError(f"Missing smoke payload: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AdaptiveSelectionSmokeError(f"Malformed smoke payload {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdaptiveSelectionSmokeError(f"Smoke payload must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def synthetic_base_fdf() -> str:
    return "\n".join(
        [
            "SystemName adaptive selection smoke",
            "SystemLabel adaptive_smoke",
            "NumberOfSpecies 1",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 6 C",
            "%endblock ChemicalSpeciesLabel",
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 8.0 0.0 0.0",
            " 0.0 8.0 0.0",
            " 0.0 0.0 8.0",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 1.0 0.0 0.0 1",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "",
        ]
    )


def write_synthetic_dataset(dataset_root: Path, *, n_test: int) -> None:
    rows = []
    for index in range(n_test):
        sample_id = f"base_{index:04d}"
        sample_dir = dataset_root / "splits" / "test" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(), encoding="utf-8")
        metadata = {
            "sample_id": sample_id,
            "material_label": "adaptive_selection_smoke",
            "system_label": "adaptive_smoke",
            "material_compatibility_hash": "smoke-material",
            "orbital_ordering_hash": "smoke-orbital",
            "neighbor_list_hash": "smoke-neighbor",
            "sparsity_pattern_hash": "smoke-sparsity",
            "basis_hash": "smoke-basis",
            "pseudopotential_hash": "smoke-pseudo",
        }
        write_json(sample_dir / "metadata.json", metadata)
        rows.append({"sample_id": sample_id, "split": "test", "sample_dir": str(sample_dir)})
    write_json(
        dataset_root / "frozen_split_manifest.json",
        {"valid": True, "split_counts": {"test": n_test}, "rows": rows},
    )


def delta_values(value: Any) -> list[float]:
    raw_values = value if isinstance(value, list) else [value]
    values = [float(item) for item in raw_values]
    if not values or any(item <= 0 for item in values):
        raise AdaptiveSelectionSmokeError("derivative.delta_ang must contain positive values.")
    return values


def validate_geometry_if_available(stencil_root: Path, *, method: str, split: str) -> dict[str, Any]:
    try:
        from validate_hamiltonian_derivative_geometry import validate_derivative_geometry_outputs
    except ModuleNotFoundError as exc:
        return {"status": "skipped_missing_dependency", "errors": 0, "reason": str(exc)}
    summary = validate_derivative_geometry_outputs(
        stencil_root,
        output_dir=stencil_root,
        method=method,
        split=split,
        require_central=True,
    )
    return {"status": "ok", "errors": int(summary["errors"])}


def run_smoke(payload: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    output_root = resolve_repo_path(payload.get("output_root") or "Comparison/results/adaptive_derivative_selection_smoke")
    if output_root.exists():
        if not overwrite:
            raise AdaptiveSelectionSmokeError(f"Output root exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    derivative = payload.get("derivative") if isinstance(payload.get("derivative"), dict) else {}
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AdaptiveSelectionSmokeError("Smoke payload must contain a non-empty cases list.")

    results = []
    for case in cases:
        if not isinstance(case, dict):
            raise AdaptiveSelectionSmokeError("Each smoke case must be an object.")
        label = str(case.get("label") or f"n_test_{case.get('n_test')}")
        n_test = int(case["n_test"])
        expected = int(case["expected_selected_base_snapshots"])
        source_root = output_root / "synthetic_datasets" / label
        stencil_root = output_root / "stencils" / label
        write_synthetic_dataset(source_root, n_test=n_test)

        manifest = build_derivative_stencils(
            source_dataset_root=source_root,
            output_stencil_root=stencil_root,
            split=str(derivative.get("base_split") or "test"),
            method=str(derivative.get("method") or "central"),
            delta_ang_values=delta_values(derivative.get("delta_ang") or 0.01),
            atom_indices_zero_based=parse_atom_indices(",".join(str(item) for item in derivative.get("atoms") or ["0"])),
            axes=parse_axes(",".join(str(item) for item in derivative.get("axes") or ["x"])),
            base_selection_policy=str(derivative.get("base_selection_policy") or "adaptive_min_fraction"),
            min_base_snapshots=int(derivative.get("min_base_snapshots") or 20),
            base_fraction=float(derivative.get("base_fraction") or 0.2),
            base_selection_seed=derivative.get("base_selection_seed"),
            overwrite=True,
        )
        validation = validate_geometry_if_available(
            stencil_root,
            method=str(derivative.get("method") or "central"),
            split=str(derivative.get("base_split") or "test"),
        )

        selected = int(manifest["selected_base_snapshot_count"])
        if int(manifest["available_base_snapshot_count"]) != n_test:
            raise AdaptiveSelectionSmokeError(f"{label}: available count mismatch.")
        if selected != expected:
            raise AdaptiveSelectionSmokeError(f"{label}: expected K={expected}, got K={selected}.")
        if manifest["base_selection_policy"] != "adaptive_min_fraction":
            raise AdaptiveSelectionSmokeError(f"{label}: wrong base_selection_policy.")
        if int(manifest["min_base_snapshots"]) != 20:
            raise AdaptiveSelectionSmokeError(f"{label}: wrong min_base_snapshots.")
        if float(manifest["base_fraction"]) != 0.2:
            raise AdaptiveSelectionSmokeError(f"{label}: wrong base_fraction.")
        if int(validation["errors"]) != 0:
            raise AdaptiveSelectionSmokeError(f"{label}: geometry validation errors.")

        results.append(
            {
                "label": label,
                "n_test": n_test,
                "selected_base_snapshot_count": selected,
                "expected_selected_base_snapshots": expected,
                "manifest": str(stencil_root / "derivative_stencil_manifest.json"),
                "validation_errors": int(validation["errors"]),
                "geometry_validation_status": validation["status"],
            }
        )

    summary = {
        "schema": "adaptive_derivative_selection_smoke_result_v1",
        "status": "ok",
        "output_root": str(output_root),
        "cases": results,
    }
    write_json(output_root / "adaptive_derivative_selection_smoke_manifest.json", summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        summary = run_smoke(read_json(args.payload), overwrite=args.overwrite)
    except AdaptiveSelectionSmokeError as exc:
        print(f"[ADAPTIVE-SELECTION-SMOKE][ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "cases": summary["cases"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
