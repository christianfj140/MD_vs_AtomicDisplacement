#!/usr/bin/env python3
"""Validate SIESTA/Graph2Mat sample bundles before training or evaluation.

Examples
--------
Validate a directory containing sample subdirectories::

    python Comparison/scripts/validate_sample_bundle.py \
      --samples-dir Comparison/workspaces/run/md/dataset/splits/train \
      --output-dir Comparison/workspaces/run/md/dataset/validation/train

Validate rows from a split manifest and use a common MD RUN.out for all rows::

    python Comparison/scripts/validate_sample_bundle.py \
      --manifest train_manifest.csv \
      --common-run-out dataset/RUN.out \
      --output-dir validation/train
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MATRIX_SUFFIXES = (".TSHS", ".HSX")
STRUCTURE_NAMES = ("RUN.fdf",)
XV_NAMES = ("siesta.XV",)


@dataclass
class SampleCandidate:
    sample_id: str
    method: str
    sample_dir: Path | None
    structure_path: Path | None
    hamiltonian_path: Path | None
    run_out_path: Path | None
    row: dict[str, Any]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_path(value: str | None, base: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def find_first(path: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def matrix_sort_key(path: Path) -> tuple[int, str]:
    numbers: list[int] = []
    for chunk in path.stem.replace("-", ".").replace("_", ".").split("."):
        if chunk.isdigit():
            numbers.append(int(chunk))
    return (numbers[-1] if numbers else 10**9, path.name)


def find_matrix(sample_dir: Path) -> Path | None:
    matrices = [
        path
        for suffix in MATRIX_SUFFIXES
        for path in sample_dir.glob(f"*{suffix}")
        if path.name != "ML_prediction.HSX"
    ]
    return sorted(matrices, key=matrix_sort_key)[0] if matrices else None


def find_matrices(sample_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for suffix in MATRIX_SUFFIXES
            for path in sample_dir.glob(f"*{suffix}")
            if path.name != "ML_prediction.HSX"
        ],
        key=matrix_sort_key,
    )


def system_label_from_fdf(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) >= 2 and parts[0].lower() == "systemlabel":
            return parts[1]
    return None


def has_atomic_coordinates(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return "%block atomiccoordinatesandatomicspecies" in text


def parse_run_out_status(path: Path | None) -> tuple[bool, bool, str]:
    if path is None or not path.exists():
        return False, False, "missing_run_out"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return False, False, f"parser_error:{exc}"
    completed = "Job completed" in text
    converged = "SCF cycle converged" in text
    if completed and converged:
        return True, True, "ok"
    missing = []
    if not completed:
        missing.append("job_completed")
    if not converged:
        missing.append("scf_converged")
    return completed, converged, "missing_" + "_and_".join(missing)


def spectral_ready(matrix_path: Path | None, require_spectral: bool) -> tuple[bool, str]:
    if not require_spectral:
        return True, "not_required"
    if matrix_path is None or not matrix_path.exists():
        return False, "missing_matrix"
    if matrix_path.suffix == ".TSHS":
        return True, "tshs_available"
    # HSX may be enough in some setups, but overlap availability is backend
    # dependent. Try sisl when installed; otherwise remain conservative.
    try:
        import sisl  # type: ignore

        sile = sisl.get_sile(str(matrix_path))
        sile.read_overlap()
        return True, "overlap_readable"
    except Exception as exc:  # pragma: no cover - environment/backend dependent.
        return False, f"overlap_unavailable:{exc}"


def candidates_from_manifest(path: Path, common_run_out: Path | None) -> list[SampleCandidate]:
    base = path.parent
    candidates = []
    for index, row in enumerate(read_csv_rows(path), start=1):
        sample_dir = as_path(row.get("sample_dir"), base)
        structure_path = as_path(row.get("structure_path"), base)
        hamiltonian_path = as_path(row.get("hamiltonian_path"), base)
        run_out_path = as_path(row.get("run_out_path") or row.get("output_path"), base) or common_run_out
        sample_id = row.get("sample_id") or (sample_dir.name if sample_dir else f"row_{index}")
        method = row.get("method") or "unknown"
        candidates.append(
            SampleCandidate(
                sample_id=sample_id,
                method=method,
                sample_dir=sample_dir,
                structure_path=structure_path,
                hamiltonian_path=hamiltonian_path,
                run_out_path=run_out_path,
                row=row,
            )
        )
    return candidates


def candidates_from_samples_dir(samples_dir: Path, method: str, common_run_out: Path | None) -> list[SampleCandidate]:
    candidates = []
    sample_dirs = sorted(path for path in samples_dir.iterdir() if path.is_dir())
    for sample_dir in sample_dirs:
        candidates.append(
            SampleCandidate(
                sample_id=sample_dir.name,
                method=method,
                sample_dir=sample_dir,
                structure_path=find_first(sample_dir, STRUCTURE_NAMES),
                hamiltonian_path=find_matrix(sample_dir),
                run_out_path=find_first(sample_dir, ("RUN.out",)) or common_run_out,
                row={},
            )
        )
    return candidates


def validate_sample(
    sample: SampleCandidate,
    *,
    require_spectral: bool,
    allow_missing_hamiltonian_debug: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    structure_path = sample.structure_path
    hamiltonian_path = sample.hamiltonian_path

    if sample.sample_dir is not None:
        if structure_path is None:
            structure_path = find_first(sample.sample_dir, STRUCTURE_NAMES)
        if hamiltonian_path is None:
            hamiltonian_path = find_matrix(sample.sample_dir)
        matrix_count = len(find_matrices(sample.sample_dir))
    else:
        matrix_count = 1 if hamiltonian_path is not None and hamiltonian_path.exists() else 0

    structure_exists = structure_path is not None and structure_path.exists()
    if not structure_exists:
        reasons.append("missing_run_fdf")
    elif not has_atomic_coordinates(structure_path):
        reasons.append("structure_missing_atomic_coordinates")

    if hamiltonian_path is None or not hamiltonian_path.exists():
        if allow_missing_hamiltonian_debug:
            reasons.append("missing_hamiltonian_debug_allowed")
        else:
            reasons.append("missing_matrix")
    if matrix_count > 1:
        reasons.append("ambiguous_reference_matrix")

    completed, converged, run_out_status = parse_run_out_status(sample.run_out_path)
    if run_out_status == "missing_run_out":
        reasons.append("missing_output")
    if run_out_status.startswith("parser_error"):
        reasons.append("parser_error")
    if not completed:
        reasons.append("job_not_completed")
    if not converged:
        reasons.append("scf_not_converged")

    spec_ok, spec_status = spectral_ready(hamiltonian_path, require_spectral)
    if not spec_ok:
        reasons.append("spectral_data_unavailable")

    label = system_label_from_fdf(structure_path) if structure_path is not None else None
    sample_id_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id)
    corresponds = True
    if label and sample_id_safe not in label and label not in sample_id_safe:
        # MD RUN.fdf files often use a generic SIESTA label; this is not fatal by
        # itself, but it is recorded because it can hide wrong sample pairing.
        corresponds = False

    is_valid = not any(
        reason
        for reason in reasons
        if reason != "missing_hamiltonian_debug_allowed"
    )
    if allow_missing_hamiltonian_debug and reasons == ["missing_hamiltonian_debug_allowed"]:
        is_valid = True

    return {
        **sample.row,
        "sample_id": sample.sample_id,
        "method": sample.method,
        "sample_dir": str(sample.sample_dir) if sample.sample_dir else "",
        "structure_path": str(structure_path) if structure_path else "",
        "hamiltonian_path": str(hamiltonian_path) if hamiltonian_path else "",
        "run_out_path": str(sample.run_out_path) if sample.run_out_path else "",
        "structure_exists": structure_exists,
        "hamiltonian_exists": hamiltonian_path is not None and hamiltonian_path.exists(),
        "job_completed": completed,
        "scf_converged": converged,
        "run_out_status": run_out_status,
        "spectral_ready": spec_ok,
        "spectral_status": spec_status,
        "system_label": label or "",
        "structure_corresponds_to_sample": corresponds,
        "status": "valid" if is_valid else "invalid",
        "invalid_reasons": ";".join(reasons),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples-dir", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--common-run-out", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-spectral", action="store_true")
    parser.add_argument("--min-valid", type=int, default=1)
    parser.add_argument("--allow-missing-hamiltonian-debug", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    common_run_out = args.common_run_out.resolve() if args.common_run_out else None
    if args.manifest:
        candidates = candidates_from_manifest(args.manifest, common_run_out)
    else:
        candidates = candidates_from_samples_dir(args.samples_dir, args.method, common_run_out)

    rows = [
        validate_sample(
            sample,
            require_spectral=args.require_spectral,
            allow_missing_hamiltonian_debug=args.allow_missing_hamiltonian_debug,
        )
        for sample in candidates
    ]
    valid_rows = [row for row in rows if row["status"] == "valid"]
    invalid_rows = [row for row in rows if row["status"] != "valid"]
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["sample_id", "status", "invalid_reasons"]

    write_csv(args.output_dir / "sample_validation_summary.csv", rows, fieldnames)
    write_csv(args.output_dir / "valid_samples.csv", valid_rows, fieldnames)
    write_csv(args.output_dir / "invalid_samples.csv", invalid_rows, fieldnames)
    summary = {
        "ok": len(valid_rows) >= args.min_valid and not invalid_rows,
        "samples_seen": len(rows),
        "valid_samples": len(valid_rows),
        "invalid_samples": len(invalid_rows),
        "min_valid": args.min_valid,
        "require_spectral": bool(args.require_spectral),
        "allow_missing_hamiltonian_debug": bool(args.allow_missing_hamiltonian_debug),
        "outputs": {
            "sample_validation_summary": str(args.output_dir / "sample_validation_summary.csv"),
            "valid_samples": str(args.output_dir / "valid_samples.csv"),
            "invalid_samples": str(args.output_dir / "invalid_samples.csv"),
            "validation_summary": str(args.output_dir / "validation_summary.json"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
