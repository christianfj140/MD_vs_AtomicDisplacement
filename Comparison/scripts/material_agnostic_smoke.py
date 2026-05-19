#!/usr/bin/env python3
"""Dry-run smoke checks for material-aware SIESTA/Graph2Mat plumbing.

This diagnostic intentionally stops before real SIESTA, Graph2Mat training, or
prediction. It validates that the repository can move a material bundle through
the lightweight layers that are safe to exercise in unit tests.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_ROOT = REPO_ROOT / "Comparison"
SHARED_DIR = REPO_ROOT / "shared"
ATOM_SCRIPTS_DIR = REPO_ROOT / "AtomDisplacement" / "scripts"
for candidate in (SHARED_DIR, ATOM_SCRIPTS_DIR, COMPARISON_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fdf_materialization import (  # noqa: E402
    FdfMaterializationError,
    extract_bundle_structure,
    materialize_sample_fdf,
)
from graph2mat_material_config import (  # noqa: E402
    apply_material_graph2mat_config,
    write_graph2mat_config_provenance,
)
from material_bundle import MaterialBundleError, file_sha256  # noqa: E402
from material_presets import resolve_material_bundle  # noqa: E402
from material_provenance import flatten_material_provenance  # noqa: E402

import generate_generic_cartesian_displacement_dataset as generic_cartesian  # noqa: E402
import generate_random_cartesian_dataset as random_cartesian  # noqa: E402


SMOKE_CASES = ("h2o", "synthetic")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def synthetic_fdf_text(*, coordinate_format: str = "Ang") -> str:
    return "\n".join(
        [
            "SystemName synthetic smoke",
            "SystemLabel synthetic_smoke",
            "NumberOfSpecies 2",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 14 Si",
            " 2 6 C",
            "%endblock ChemicalSpeciesLabel",
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 6.0 0.0 0.0",
            " 0.0 6.0 0.0",
            " 0.0 0.0 6.0",
            "%endblock LatticeVectors",
            f"AtomicCoordinatesFormat {coordinate_format}",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 2.0 0.0 0.0 2",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "MeshCutoff 200 Ry",
            "Save.HS T",
            "TS.HS.Save T",
            "TS.DE.Save T",
            "XML.Write T",
            "",
        ]
    )


def write_synthetic_material(
    root: Path,
    *,
    missing_pseudo: str | None = None,
    coordinate_format: str = "Ang",
    include_basis: bool = True,
) -> dict[str, Any]:
    material_root = root / "materials" / "sic"
    material_root.mkdir(parents=True, exist_ok=True)
    (material_root / "RUN.fdf").write_text(
        synthetic_fdf_text(coordinate_format=coordinate_format),
        encoding="utf-8",
    )
    pseudo_dir = material_root / "pseudos"
    pseudo_dir.mkdir(exist_ok=True)
    pseudo_payload = {"Si": "si pseudo\n", "C": "c pseudo\n"}
    for label, content in pseudo_payload.items():
        if label == missing_pseudo:
            continue
        extension = ".psf" if label == "Si" else ".psml"
        (pseudo_dir / f"{label}{extension}").write_text(content, encoding="utf-8")
    basis_dir = material_root / "basis"
    basis_dir.mkdir(exist_ok=True)
    if include_basis:
        (basis_dir / "Si.ion.xml").write_text("<ion symbol=\"Si\" />\n", encoding="utf-8")
        (basis_dir / "C.ion.xml").write_text("<ion symbol=\"C\" />\n", encoding="utf-8")
    return {
        "material": {
            "label": "sic",
            "fdf": "materials/sic/RUN.fdf",
            "pseudopotential_dir": "materials/sic/pseudos",
            "basis_dir": "materials/sic/basis",
            "structure_type": "crystal",
        }
    }


def material_config_for_case(case: str, work_root: Path) -> tuple[dict[str, Any], Path]:
    if case == "h2o":
        return {"material": {"preset": "h2o"}}, REPO_ROOT
    if case == "synthetic":
        return write_synthetic_material(work_root), work_root
    raise RuntimeError(f"Unsupported smoke case: {case}")


def graph2mat_config(material_config: dict[str, Any]) -> dict[str, Any]:
    return {
        **json.loads(json.dumps(material_config)),
        "training": {
            "data": {
                "out_matrix": "hamiltonian",
                "symmetric_matrix": True,
                "matrix_component_policy": "h_only",
                "n_matrix_components": 1,
                "basis_files": "unset",
                "train_runs": "../dataset/splits/train/*/RUN.fdf",
                "val_runs": "../dataset/splits/validation/*/RUN.fdf",
            },
            "model": {},
            "trainer": {},
        },
        "testing": {
            "data": {
                "out_matrix": "hamiltonian",
                "symmetric_matrix": True,
                "matrix_component_policy": "h_only",
                "n_matrix_components": 1,
                "basis_files": "unset",
                "test_runs": "../dataset/splits/test/*/RUN.fdf",
            }
        },
        "prediction": {
            "data": {
                "out_matrix": "hamiltonian",
                "symmetric_matrix": True,
                "matrix_component_policy": "h_only",
                "n_matrix_components": 1,
                "basis_files": "unset",
            }
        },
    }


def write_split_fixture(dataset_dir: Path, source_run_fdf: Path) -> dict[str, str]:
    split_hashes: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        sample_dir = dataset_dir / "splits" / split_name / "sample_000"
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_fdf = sample_dir / "RUN.fdf"
        shutil.copy2(source_run_fdf, target_fdf)
        manifest_path = dataset_dir / "splits" / f"{split_name}_manifest.csv"
        manifest_path.write_text(
            "sample_id,sample_dir,structure_path,valid,status\n"
            f"{split_name}_000,{sample_dir},{target_fdf},True,valid\n",
            encoding="utf-8",
        )
        split_hashes[split_name] = file_sha256(manifest_path)
    return split_hashes


def materialize_reference_sample(
    *,
    validated: Any,
    case_root: Path,
) -> dict[str, Any]:
    structure = extract_bundle_structure(validated)
    output_fdf = case_root / "materialized_reference" / "RUN.fdf"
    materialized = materialize_sample_fdf(
        validated.bundle.fdf,
        output_fdf,
        positions_ang=structure.positions_ang,
        atom_species=structure.atom_species,
        lattice_vectors_ang=structure.lattice_vectors_ang,
        system_label=f"{validated.bundle.label}_smoke_reference",
        system_name=f"{validated.bundle.label} smoke reference",
        structure_type=validated.bundle.structure_type,
    )
    return {
        "structure": structure,
        "path": output_fdf,
        "metadata": materialized.metadata,
    }


def run_graph2mat_dry_config(
    *,
    material_config: dict[str, Any],
    material_base_dir: Path,
    case_root: Path,
    source_run_fdf: Path,
) -> dict[str, Any]:
    dataset_dir = case_root / "graph2mat_dataset"
    training_dir = case_root / "graph2mat_training"
    training_dir.mkdir(parents=True, exist_ok=True)
    split_hashes = write_split_fixture(dataset_dir, source_run_fdf)
    config = graph2mat_config(material_config)
    provenance = apply_material_graph2mat_config(
        config,
        base_dir=material_base_dir,
        dataset_dir=dataset_dir,
        training_dir=training_dir,
    )
    config_path = training_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    provenance_path = write_graph2mat_config_provenance(
        config_path,
        provenance,
        validation_metadata={"validation_source": "training.data.val_runs"},
    )
    return {
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "provenance_path": str(provenance_path),
        "basis_files": config["training"]["data"]["basis_files"],
        "split_hashes": split_hashes,
        "provenance": provenance,
    }


def run_material_case(
    case: str,
    *,
    output_root: Path,
) -> dict[str, Any]:
    case_root = output_root / case
    case_root.mkdir(parents=True, exist_ok=True)
    material_config, material_base_dir = material_config_for_case(case, case_root)
    resolved = resolve_material_bundle(material_config, base_dir=material_base_dir)
    validated = resolved.validated
    reference = materialize_reference_sample(validated=validated, case_root=case_root)
    structure = reference["structure"]

    cartesian_config = {
        **json.loads(json.dumps(material_config)),
        "generation": {"sample_id_format": "cart_{index:04d}"},
        "atomic_displacement": {
            "recipe": "generic_cartesian",
            "amplitude_ang": 0.01,
            "selected_species": None,
            "include_base": False,
        },
    }
    cartesian_manifest = generic_cartesian.generate_dataset(
        cartesian_config,
        output_dir=case_root / "generic_cartesian_samples",
        base_dir=material_base_dir,
        overwrite=True,
    )

    random_config = {
        **json.loads(json.dumps(material_config)),
        "dataset_recipe": {
            "recipe_id": f"{case}_smoke_random",
            "recipe_label": f"{case} smoke random",
        },
        "random_cartesian": {
            "recipe": "generic_cartesian_noise",
            "n_structures": 3,
            "max_displacement_ang": 0.005,
            "distribution": "uniform",
            "selected_species": None,
            "min_interatomic_distance_ang": 0.3,
            "remove_center_of_mass_translation": False,
            "seed": 12345,
            "variants_per_family": 1,
            "max_attempts_per_structure": 20,
        },
    }
    with contextlib.redirect_stdout(sys.stderr):
        random_manifest = random_cartesian.generate_dataset(
            random_config,
            output_dir=case_root / "random_cartesian_samples",
            material_base_dir=material_base_dir,
        )
        random_repeat_manifest = random_cartesian.generate_dataset(
            random_config,
            output_dir=case_root / "random_cartesian_samples_repeat",
            material_base_dir=material_base_dir,
        )
    random_deterministic = (
        random_manifest["deterministic_hashes"]["sample_family_hashes"]
        == random_repeat_manifest["deterministic_hashes"]["sample_family_hashes"]
        and random_manifest["siesta_input_hashes"] == random_repeat_manifest["siesta_input_hashes"]
    )
    if not random_deterministic:
        raise RuntimeError(f"{case}: generic random Cartesian generation is not deterministic.")

    graph2mat = run_graph2mat_dry_config(
        material_config=material_config,
        material_base_dir=material_base_dir,
        case_root=case_root,
        source_run_fdf=reference["path"],
    )
    provenance = flatten_material_provenance(
        {"material": resolved.to_manifest_dict(), "material_atom_count": structure.atom_count},
        cartesian_manifest,
        random_manifest,
        graph2mat["provenance"],
    )
    if not provenance.get("material_label") or not provenance.get("fdf_sha256"):
        raise RuntimeError(f"{case}: material provenance is incomplete.")

    report = {
        "ok": True,
        "case": case,
        "external_execution_required": False,
        "material": {
            "label": validated.bundle.label,
            "source": resolved.source,
            "preset": resolved.preset,
            "structure_type": validated.bundle.structure_type,
            "species": [item.to_dict() for item in validated.species],
            "atom_count": structure.atom_count,
            "fdf_sha256": resolved.to_manifest_dict().get("fdf_sha256"),
            "pseudopotential_sha256": resolved.to_manifest_dict().get("pseudopotential_sha256"),
        },
        "fdf_materialization": {
            "run_fdf": str(reference["path"]),
            "metadata": reference["metadata"],
        },
        "generic_cartesian": {
            "generated_structures": cartesian_manifest["generated_structures"],
            "sample_root": cartesian_manifest["sample_root"],
            "first_sample": cartesian_manifest["samples"][0],
            "material_label": cartesian_manifest["material"]["label"],
        },
        "generic_random_cartesian": {
            "generated_structures": random_manifest["generated_structures"],
            "sample_root": random_manifest["dataset_root"],
            "split_strategy": random_manifest["split_strategy"],
            "split_group_key": random_manifest["split_group_key"],
            "deterministic": random_deterministic,
            "material_label": random_manifest["material"]["label"],
        },
        "graph2mat_config": {
            "config_path": graph2mat["config_path"],
            "config_sha256": graph2mat["config_sha256"],
            "basis_files": graph2mat["basis_files"],
            "provenance_path": graph2mat["provenance_path"],
            "matrix_target": graph2mat["provenance"]["graph2mat"]["matrix_target"],
            "matrix_component_policy": graph2mat["provenance"]["graph2mat"]["matrix_component_policy"],
            "n_matrix_components": graph2mat["provenance"]["graph2mat"]["n_matrix_components"],
            "split_hashes": graph2mat["split_hashes"],
        },
        "material_provenance": provenance,
    }
    write_json(case_root / "smoke_manifest.json", report)
    return report


def expected_failure_checks(output_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    missing_root = output_root / "expected_failure_missing_pseudo"
    config = write_synthetic_material(missing_root, missing_pseudo="C")
    try:
        resolve_material_bundle(config, base_dir=missing_root)
    except MaterialBundleError as exc:
        checks.append({"case": "missing_pseudo", "ok": "true", "message": str(exc)})
    else:
        raise RuntimeError("Expected missing pseudopotential fixture to fail.")

    unsupported_root = output_root / "expected_failure_unsupported_fdf"
    config = write_synthetic_material(unsupported_root, coordinate_format="Fractional")
    try:
        resolved = resolve_material_bundle(config, base_dir=unsupported_root)
        extract_bundle_structure(resolved.validated)
    except FdfMaterializationError as exc:
        checks.append({"case": "unsupported_fdf", "ok": "true", "message": str(exc)})
    else:
        raise RuntimeError("Expected unsupported FDF coordinate format to fail.")
    return checks


def run_smoke(
    *,
    cases: list[str],
    output_root: Path,
    include_failure_checks: bool = True,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reports = [run_material_case(case, output_root=output_root) for case in cases]
    failures = expected_failure_checks(output_root) if include_failure_checks else []
    payload = {
        "ok": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "external_execution_required": False,
        "output_root": str(output_root),
        "cases": reports,
        "expected_failure_checks": failures,
    }
    write_json(output_root / "material_agnostic_smoke_report.json", payload)
    return payload


def parse_cases(value: str) -> list[str]:
    if value == "both":
        return list(SMOKE_CASES)
    if value not in SMOKE_CASES:
        raise RuntimeError(f"--case debe ser h2o, synthetic o both; recibido {value!r}.")
    return [value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run material-agnostic dry-run smoke checks without SIESTA or Graph2Mat execution."
    )
    parser.add_argument("--case", choices=("h2o", "synthetic", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failure-checks", action="store_true")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="When --output-dir is omitted, keep the generated temporary directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = parse_cases(args.case)
    temp_context: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir is None:
        if args.keep_temp:
            output_root = COMPARISON_ROOT / "results" / f"material_agnostic_smoke_{datetime.now():%Y%m%d_%H%M%S}"
        else:
            temp_context = tempfile.TemporaryDirectory(prefix="material_agnostic_smoke_")
            output_root = Path(temp_context.name)
    else:
        output_root = args.output_dir
        if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {output_root}. Use --overwrite or choose a fresh path."
            )
        if output_root.exists() and args.overwrite:
            shutil.rmtree(output_root)
    try:
        report = run_smoke(
            cases=cases,
            output_root=output_root,
            include_failure_checks=not args.skip_failure_checks,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
