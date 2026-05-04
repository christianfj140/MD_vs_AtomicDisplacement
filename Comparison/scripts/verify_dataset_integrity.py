#!/usr/bin/env python3
"""Validate MD and AtomicDisplacement datasets used by the comparison pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from validate_sample_bundle import SampleCandidate, find_first, find_matrix, validate_sample


REPO_ROOT = Path(__file__).resolve().parents[2]


def numeric_step_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_steps(root: Path, label: str, *, strict_validation: bool) -> dict[str, Any]:
    steps = numeric_step_dirs(root)
    rows = []
    for step in steps:
        sample = SampleCandidate(
            sample_id=step.name,
            method=label.lower(),
            sample_dir=step,
            structure_path=find_first(step, ("RUN.fdf",)),
            hamiltonian_path=find_matrix(step),
            run_out_path=find_first(step, ("RUN.out",)),
            row={},
        )
        rows.append(
            validate_sample(
                sample,
                require_spectral=False,
                allow_missing_hamiltonian_debug=not strict_validation,
            )
        )
    invalid = [row for row in rows if row["status"] != "valid"]
    return {
        "label": label,
        "root": str(root),
        "steps": len(steps),
        "valid_samples": sum(1 for row in rows if row["status"] == "valid"),
        "invalid_samples": len(invalid),
        "missing_run_fdf": [row["sample_id"] for row in rows if "missing_run_fdf" in row["invalid_reasons"]],
        "missing_matrix": [row["sample_id"] for row in rows if "missing_matrix" in row["invalid_reasons"]],
        "missing_output": [row["sample_id"] for row in rows if "missing_output" in row["invalid_reasons"]],
        "unconverged_samples": [row["sample_id"] for row in rows if "scf_not_converged" in row["invalid_reasons"]],
        "invalid_reasons": {row["sample_id"]: row["invalid_reasons"] for row in invalid},
        "ok": bool(steps) and not invalid,
    }


def expected_fc_count(first_atom: int, last_atom: int, include_reference: bool) -> int:
    displaced = last_atom - first_atom + 1
    return (1 if include_reference else 0) + 6 * displaced


def fc_run_labels(manifest: dict[str, Any]) -> list[str]:
    runs = manifest.get("runs")
    if isinstance(runs, list):
        labels = []
        for run in runs:
            if isinstance(run, dict) and run.get("label"):
                labels.append(str(run["label"]))
        return labels
    return []


def expected_normalized_samples(manifest: dict[str, Any], legacy_expected: int) -> int | None:
    selected = manifest.get("selected_sample_ids")
    if isinstance(selected, list):
        return len(selected)
    requested = manifest.get("requested_structures")
    if isinstance(requested, int):
        return requested
    total = 0
    has_count = False
    for run in manifest.get("runs") or []:
        if isinstance(run, dict) and isinstance(run.get("selected_count"), int):
            total += int(run["selected_count"])
            has_count = True
    if has_count:
        return total
    if manifest.get("generation_mode") == "siesta_fc_multi_run":
        return None
    return legacy_expected


def collect_fc_outputs(raw_dir: Path) -> list[Path]:
    outputs = sorted(raw_dir.glob("*.FC")) + sorted(raw_dir.glob("FC"))
    fc_runs = raw_dir / "FC_runs"
    if fc_runs.exists():
        for run_dir in sorted(path for path in fc_runs.iterdir() if path.is_dir()):
            outputs.extend(sorted(run_dir.glob("*.FC")))
            outputs.extend(sorted(run_dir.glob("FC")))
    return sorted(set(outputs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify H2O MD_steps and FC_steps/AtDis_steps.")
    parser.add_argument("--base-fdf", type=Path, default=REPO_ROOT / "AtomDisplacement" / "base" / "RUN.fdf")
    parser.add_argument("--md-steps-dir", type=Path, default=REPO_ROOT / "MD" / "dataset" / "MD_steps")
    parser.add_argument("--fc-steps-dir", type=Path, default=REPO_ROOT / "AtomDisplacement" / "dataset" / "FC_steps")
    parser.add_argument("--fc-raw-dir", type=Path, default=REPO_ROOT / "AtomDisplacement" / "dataset")
    parser.add_argument(
        "--samples-manifest",
        type=Path,
        default=REPO_ROOT / "AtomDisplacement" / "dataset" / "samples_manifest.json",
    )
    parser.add_argument("--siesta-bin", default="siesta")
    parser.add_argument("--fc-first", type=int, default=1)
    parser.add_argument("--fc-last", type=int, default=3)
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "Comparison" / "results" / "dataset_integrity.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors: list[str] = []
    if not args.base_fdf.exists():
        errors.append(f"No existe RUN.fdf base: {args.base_fdf}")
    if shutil.which(args.siesta_bin) is None:
        errors.append(f"No se encontro ejecutable SIESTA en PATH: {args.siesta_bin}")
    if args.fc_first < 1 or args.fc_last < args.fc_first:
        errors.append(f"Rango FC invalido: {args.fc_first}-{args.fc_last}")

    manifest = read_json(args.samples_manifest)
    raw_labels = fc_run_labels(manifest)
    detected_raw_labels: list[str] = []
    fc_runs_dir = args.fc_raw_dir / "FC_runs"
    if fc_runs_dir.exists():
        detected_raw_labels = sorted(path.name for path in fc_runs_dir.iterdir() if path.is_dir())
    missing_raw_labels = sorted(set(raw_labels) - set(detected_raw_labels))

    md = validate_steps(args.md_steps_dir, "MD", strict_validation=args.strict_validation)
    fc = validate_steps(args.fc_steps_dir, "FC", strict_validation=args.strict_validation)
    legacy_fc_expected = expected_fc_count(args.fc_first, args.fc_last, args.include_reference)
    normalized_expected = expected_normalized_samples(manifest, legacy_fc_expected)
    fc_outputs = collect_fc_outputs(args.fc_raw_dir)
    pseudo_files = sorted(args.base_fdf.parent.glob("*.psf"))

    if not pseudo_files:
        errors.append(f"No se encontraron pseudopotenciales .psf junto a {args.base_fdf}")
    if raw_labels and missing_raw_labels:
        errors.append(f"FC_runs esperados no encontrados: {missing_raw_labels}")
    if normalized_expected is not None and fc["steps"] < normalized_expected:
        errors.append(
            f"FC_steps contiene {fc['steps']} samples, esperado al menos {normalized_expected}."
        )
    if not raw_labels and normalized_expected is not None and fc["steps"] != normalized_expected:
        errors.append(
            f"FC genero {fc['steps']} steps, esperado {normalized_expected} "
            f"para atomos {args.fc_first}-{args.fc_last}."
        )
    if not fc_outputs:
        errors.append(f"No se encontro archivo .FC en {args.fc_raw_dir} ni en FC_runs/*")
    for result in (md, fc):
        if result["missing_run_fdf"]:
            errors.append(f"{result['label']}: steps sin RUN.fdf: {result['missing_run_fdf']}")
        if result["missing_matrix"]:
            errors.append(f"{result['label']}: steps sin Hamiltoniano .TSHS/.HSX: {result['missing_matrix']}")
        if args.strict_validation and result["missing_output"]:
            errors.append(f"{result['label']}: steps sin RUN.out: {result['missing_output']}")
        if args.strict_validation and result["unconverged_samples"]:
            errors.append(f"{result['label']}: steps sin SCF convergido: {result['unconverged_samples']}")

    report = {
        "ok": not errors,
        "errors": errors,
        "siesta_bin": args.siesta_bin,
        "strict_validation": bool(args.strict_validation),
        "base_fdf": str(args.base_fdf),
        "pseudopotentials": [str(path) for path in pseudo_files],
        "md": md,
        "fc": fc,
        "fc_mode": manifest.get("generation_mode") or ("multi_amplitude" if raw_labels else "legacy_single_amplitude"),
        "fc_expected_legacy_steps": legacy_fc_expected,
        "expected_raw_fc_runs": raw_labels,
        "detected_raw_fc_runs": detected_raw_labels,
        "missing_raw_fc_runs": missing_raw_labels,
        "expected_normalized_samples": normalized_expected,
        "actual_normalized_samples": fc["steps"],
        "fc_outputs": [str(path) for path in fc_outputs],
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] Reporte escrito en {args.output}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
