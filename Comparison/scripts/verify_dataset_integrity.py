#!/usr/bin/env python3
"""Validate MD and SIESTA-FC datasets used by the comparison pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def numeric_step_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def has_matrix(step: Path) -> bool:
    return any(step.glob("*.TSHS")) or any(step.glob("*.HSX"))


def check_steps(root: Path, label: str) -> dict[str, Any]:
    steps = numeric_step_dirs(root)
    missing_run = [step.name for step in steps if not (step / "RUN.fdf").exists()]
    missing_matrix = [step.name for step in steps if not has_matrix(step)]
    return {
        "label": label,
        "root": str(root),
        "steps": len(steps),
        "missing_run_fdf": missing_run,
        "missing_hamiltonian": missing_matrix,
        "ok": bool(steps) and not missing_run and not missing_matrix,
    }


def expected_fc_count(first_atom: int, last_atom: int, include_reference: bool) -> int:
    displaced = last_atom - first_atom + 1
    return (1 if include_reference else 0) + 6 * displaced


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify H2O MD_steps and FC_steps/AtDis_steps.")
    parser.add_argument("--base-fdf", type=Path, default=REPO_ROOT / "AtomDisplacement" / "base" / "RUN.fdf")
    parser.add_argument("--md-steps-dir", type=Path, default=REPO_ROOT / "MD" / "dataset" / "MD_steps")
    parser.add_argument("--fc-steps-dir", type=Path, default=REPO_ROOT / "AtomDisplacement" / "dataset" / "FC_steps")
    parser.add_argument("--fc-raw-dir", type=Path, default=REPO_ROOT / "AtomDisplacement" / "dataset")
    parser.add_argument("--siesta-bin", default="siesta")
    parser.add_argument("--fc-first", type=int, default=1)
    parser.add_argument("--fc-last", type=int, default=3)
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
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

    md = check_steps(args.md_steps_dir, "MD")
    fc = check_steps(args.fc_steps_dir, "FC")
    fc_expected = expected_fc_count(args.fc_first, args.fc_last, args.include_reference)
    fc_files = sorted(args.fc_raw_dir.glob("*.FC")) + sorted(args.fc_raw_dir.glob("FC"))
    pseudo_files = sorted(args.base_fdf.parent.glob("*.psf"))

    if not pseudo_files:
        errors.append(f"No se encontraron pseudopotenciales .psf junto a {args.base_fdf}")
    if fc["steps"] != fc_expected:
        errors.append(
            f"FC genero {fc['steps']} steps, esperado {fc_expected} "
            f"para atomos {args.fc_first}-{args.fc_last}."
        )
    if not fc_files:
        errors.append(f"No se encontro archivo .FC en {args.fc_raw_dir}")
    for result in (md, fc):
        if result["missing_run_fdf"]:
            errors.append(f"{result['label']}: steps sin RUN.fdf: {result['missing_run_fdf']}")
        if result["missing_hamiltonian"]:
            errors.append(
                f"{result['label']}: steps sin Hamiltoniano .TSHS/.HSX: "
                f"{result['missing_hamiltonian']}"
            )

    report = {
        "ok": not errors,
        "errors": errors,
        "siesta_bin": args.siesta_bin,
        "base_fdf": str(args.base_fdf),
        "pseudopotentials": [str(path) for path in pseudo_files],
        "md": md,
        "fc": fc,
        "fc_expected_steps": fc_expected,
        "fc_outputs": [str(path) for path in fc_files],
    }

    print(json.dumps(report, indent=2))
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Reporte escrito en {args.output}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
