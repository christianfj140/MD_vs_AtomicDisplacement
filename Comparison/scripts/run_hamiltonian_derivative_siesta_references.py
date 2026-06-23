#!/usr/bin/env python3
"""Run or stage SIESTA Hamiltonian references for derivative stencil structures."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reference_selection import choose_reference_matrix, file_sha256  # noqa: E402


FORBIDDEN_REFERENCE_NAMES = {"ML_prediction.HSX"}
STATUS_FIELDS = [
    "sample_id",
    "status",
    "structure_dir",
    "reference_dir",
    "reference_matrix",
    "reference_matrix_sha256",
    "command",
    "returncode",
    "started_at",
    "finished_at",
    "error",
]
STRUCTURE_SKIP_SUFFIXES = {".HSX", ".TSHS", ".TSDE", ".nc", ".out", ".XV", ".STRUCT_OUT", ".ORB_INDX"}
PSEUDOPOTENTIAL_SUFFIXES = (".psf", ".vps", ".psml")
PSEUDOPOTENTIAL_DIR_CANDIDATES = ("", "pseudopotentials", "pseudos")


class DerivativeSiestaReferenceError(RuntimeError):
    """Raised when derivative SIESTA reference staging fails closed."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DerivativeSiestaReferenceError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DerivativeSiestaReferenceError(f"Malformed JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DerivativeSiestaReferenceError(f"JSON payload must be an object: {path}")
    return payload


def discover_structure_samples(stencil_root: Path) -> list[Path]:
    structures_root = stencil_root / "structures"
    if not structures_root.exists():
        raise DerivativeSiestaReferenceError(f"Missing derivative stencil structures directory: {structures_root}")
    return sorted(path for path in structures_root.iterdir() if path.is_dir())


def clean_reference_dir(reference_dir: Path, *, overwrite: bool) -> None:
    if reference_dir.exists() and overwrite:
        shutil.rmtree(reference_dir)


def copy_structure_inputs(structure_dir: Path, reference_dir: Path) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(structure_dir.iterdir()):
        if not source.is_file():
            continue
        if source.name == "metadata.json":
            continue
        if source.name in FORBIDDEN_REFERENCE_NAMES:
            continue
        if source.suffix in STRUCTURE_SKIP_SUFFIXES:
            continue
        shutil.copy2(source, reference_dir / source.name)
    run_fdf = structure_dir / "RUN.fdf"
    if not run_fdf.exists():
        raise DerivativeSiestaReferenceError(f"Missing RUN.fdf for derivative stencil sample: {structure_dir}")
    shutil.copy2(run_fdf, reference_dir / "RUN.fdf")
    metadata = structure_dir / "metadata.json"
    if metadata.exists():
        shutil.copy2(metadata, reference_dir / "metadata.json")


def copy_existing_reference(sample_id: str, reference_dir: Path, existing_reference_root: Path | None) -> Path | None:
    if existing_reference_root is None:
        return None
    source_dir = existing_reference_root / sample_id
    selection = choose_reference_matrix(source_dir)
    if not selection.ok or selection.path is None:
        return None
    reference_dir.mkdir(parents=True, exist_ok=True)
    target = reference_dir / selection.path.name
    shutil.copy2(selection.path, target)
    return target


def resolve_source_dataset_root(
    *,
    stencil_root: Path,
    source_dataset_root: Path | None,
) -> Path | None:
    if source_dataset_root is not None:
        if not source_dataset_root.exists():
            raise DerivativeSiestaReferenceError(
                f"Configured source dataset root does not exist: {source_dataset_root}"
            )
        return source_dataset_root
    manifest_path = stencil_root / "derivative_stencil_manifest.json"
    if not manifest_path.exists():
        return None
    payload = read_json(manifest_path)
    raw = str(payload.get("source_dataset_root") or "").strip()
    if not raw:
        return None
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve(strict=False)
    if not resolved.exists():
        raise DerivativeSiestaReferenceError(
            f"Derivative stencil manifest points to a missing source dataset root: {resolved} "
            f"(from {manifest_path})"
        )
    return resolved


def required_species_labels(run_fdf: Path) -> list[str]:
    labels: list[str] = []
    in_block = False
    for raw_line in run_fdf.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        lowered = line.lower()
        if lowered == "%block chemicalspecieslabel":
            in_block = True
            continue
        if lowered == "%endblock chemicalspecieslabel":
            break
        if not in_block or not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            label = parts[2].strip()
            if label and label not in labels:
                labels.append(label)
    if not labels:
        raise DerivativeSiestaReferenceError(
            f"Could not determine required pseudopotential species from {run_fdf}"
        )
    return labels


def stage_required_pseudopotentials(
    *,
    reference_dir: Path,
    source_dataset_root: Path,
) -> list[str]:
    species_labels = required_species_labels(reference_dir / "RUN.fdf")
    search_roots = [source_dataset_root / relative for relative in PSEUDOPOTENTIAL_DIR_CANDIDATES]
    staged: list[str] = []
    for label in species_labels:
        candidates = [
            root / f"{label}{suffix}"
            for root in search_roots
            for suffix in PSEUDOPOTENTIAL_SUFFIXES
        ]
        source = next((candidate for candidate in candidates if candidate.exists()), None)
        if source is None:
            raise DerivativeSiestaReferenceError(
                "Missing required pseudopotential for species "
                f"{label!r} in derivative SIESTA calculation directory {reference_dir}. "
                f"Searched dataset/source root {source_dataset_root}. "
                "Candidate filenames tried: "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        target = reference_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
        staged.append(target.name)
    return staged


def run_siesta(reference_dir: Path, *, command: str, use_shell: bool = False) -> dict[str, Any]:
    run_fdf = reference_dir / "RUN.fdf"
    run_out = reference_dir / "RUN.out"
    command_args: str | list[str] = command if use_shell else shlex.split(command)
    with run_fdf.open("r", encoding="utf-8") as stdin, run_out.open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(
            command_args,
            cwd=reference_dir,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            shell=use_shell,
            check=False,
            text=True,
        )
    return {
        "command": command if use_shell else command_args,
        "shell": use_shell,
        "returncode": completed.returncode,
        "stdout_path": str(run_out),
    }


def sample_row(
    *,
    sample_id: str,
    status: str,
    structure_dir: Path,
    reference_dir: Path,
    command: str | None = None,
    returncode: int | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    selection = choose_reference_matrix(reference_dir)
    reference_matrix = selection.path if selection.ok else None
    return {
        "sample_id": sample_id,
        "status": status,
        "structure_dir": str(structure_dir),
        "reference_dir": str(reference_dir),
        "reference_matrix": str(reference_matrix) if reference_matrix else "",
        "reference_matrix_sha256": file_sha256(reference_matrix),
        "command": command or "",
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
        "reference_selection": {
            "ok": selection.ok,
            "reason": selection.reason,
            "candidates": list(selection.candidates),
        },
    }


def _process_siesta_reference_sample(
    *,
    structure_dir: Path,
    output_reference_root: Path,
    existing_reference_root: Path | None,
    resolved_source_dataset_root: Path | None,
    siesta_command: str,
    overwrite: bool,
    skip_if_exists: bool,
    siesta_shell: bool,
) -> dict[str, Any]:
    sample_id = structure_dir.name
    reference_dir = output_reference_root / sample_id
    started_at = time.time()
    try:
        if not (structure_dir / "RUN.fdf").exists():
            return sample_row(
                sample_id=sample_id,
                status="error",
                structure_dir=structure_dir,
                reference_dir=reference_dir,
                started_at=started_at,
                finished_at=time.time(),
                error="missing_structure_run_fdf",
            )
        existing_selection = choose_reference_matrix(reference_dir)
        if existing_selection.ok and skip_if_exists and not overwrite:
            return sample_row(
                sample_id=sample_id,
                status="skipped_existing",
                structure_dir=structure_dir,
                reference_dir=reference_dir,
                started_at=started_at,
                finished_at=time.time(),
            )
        clean_reference_dir(reference_dir, overwrite=overwrite)
        copy_structure_inputs(structure_dir, reference_dir)
        staged = copy_existing_reference(sample_id, reference_dir, existing_reference_root)
        if staged is not None:
            return sample_row(
                sample_id=sample_id,
                status="staged",
                structure_dir=structure_dir,
                reference_dir=reference_dir,
                started_at=started_at,
                finished_at=time.time(),
            )
        if resolved_source_dataset_root is None:
            raise DerivativeSiestaReferenceError(
                "Derivative SIESTA references require a source dataset root to stage pseudopotentials. "
                f"Calculation directory: {reference_dir}. "
                "Provide --source-dataset-root or ensure derivative_stencil_manifest.json records source_dataset_root."
            )
        stage_required_pseudopotentials(
            reference_dir=reference_dir,
            source_dataset_root=resolved_source_dataset_root,
        )
        run_record = run_siesta(reference_dir, command=siesta_command, use_shell=siesta_shell)
        selection = choose_reference_matrix(reference_dir)
        status = "ok" if run_record["returncode"] == 0 and selection.ok else "error"
        error = "" if status == "ok" else selection.reason if run_record["returncode"] == 0 else "siesta_returncode_nonzero"
        return sample_row(
            sample_id=sample_id,
            status=status,
            structure_dir=structure_dir,
            reference_dir=reference_dir,
            command=_command_display(run_record["command"]),
            returncode=int(run_record["returncode"]),
            started_at=started_at,
            finished_at=time.time(),
            error=error,
        )
    except Exception as exc:
        return sample_row(
            sample_id=sample_id,
            status="error",
            structure_dir=structure_dir,
            reference_dir=reference_dir,
            command=siesta_command,
            started_at=started_at,
            finished_at=time.time(),
            error=str(exc),
        )


def _validate_workers(workers: int) -> int:
    if isinstance(workers, bool):
        raise DerivativeSiestaReferenceError("workers must be a positive integer.")
    try:
        value = int(workers)
    except (TypeError, ValueError) as exc:
        raise DerivativeSiestaReferenceError("workers must be a positive integer.") from exc
    if value < 1 or (isinstance(workers, float) and workers != value):
        raise DerivativeSiestaReferenceError("workers must be a positive integer.")
    return value


def run_derivative_siesta_references(
    *,
    stencil_root: Path,
    output_reference_root: Path | None = None,
    source_dataset_root: Path | None = None,
    existing_reference_root: Path | None = None,
    siesta_command: str = "siesta",
    overwrite: bool = False,
    skip_if_exists: bool = True,
    diagnostic_only: bool = False,
    workers: int = 1,
    max_samples: int | None = None,
    max_jobs: int | None = None,
    max_jobs_alias_used: bool = False,
    siesta_shell: bool = False,
) -> dict[str, Any]:
    output_reference_root = output_reference_root or stencil_root / "siesta_hamiltonians"
    validated_workers = _validate_workers(workers)
    structures = discover_structure_samples(stencil_root)
    resolved_source_dataset_root = (
        None
        if existing_reference_root is not None
        else resolve_source_dataset_root(
            stencil_root=stencil_root,
            source_dataset_root=source_dataset_root,
        )
    )
    effective_max_samples = _effective_max_samples(max_samples=max_samples, max_jobs=max_jobs)
    if effective_max_samples is not None:
        structures = structures[: max(0, int(effective_max_samples))]
    if validated_workers == 1 or len(structures) < 2:
        rows = [
            _process_siesta_reference_sample(
                structure_dir=structure_dir,
                output_reference_root=output_reference_root,
                existing_reference_root=existing_reference_root,
                resolved_source_dataset_root=resolved_source_dataset_root,
                siesta_command=siesta_command,
                overwrite=overwrite,
                skip_if_exists=skip_if_exists,
                siesta_shell=siesta_shell,
            )
            for structure_dir in structures
        ]
    else:
        with ThreadPoolExecutor(max_workers=validated_workers) as executor:
            futures = [
                executor.submit(
                    _process_siesta_reference_sample,
                    structure_dir=structure_dir,
                    output_reference_root=output_reference_root,
                    existing_reference_root=existing_reference_root,
                    resolved_source_dataset_root=resolved_source_dataset_root,
                    siesta_command=siesta_command,
                    overwrite=overwrite,
                    skip_if_exists=skip_if_exists,
                    siesta_shell=siesta_shell,
                )
                for structure_dir in structures
            ]
            rows = [future.result() for future in futures]

    failed = [row for row in rows if row["status"] == "error"]
    manifest = {
        "schema_version": "derivative_siesta_references_v1",
        "stencil_root": str(stencil_root),
        "structures_root": str(stencil_root / "structures"),
        "output_reference_root": str(output_reference_root),
        "source_dataset_root": str(resolved_source_dataset_root) if resolved_source_dataset_root else "",
        "existing_reference_root": str(existing_reference_root) if existing_reference_root else "",
        "siesta_hamiltonians_root": str(output_reference_root),
        "siesta_command": siesta_command if existing_reference_root is None else "",
        "overwrite": overwrite,
        "skip_if_exists": skip_if_exists,
        "diagnostic_only": diagnostic_only,
        "workers": validated_workers,
        "parallel_execution_enabled": validated_workers > 1,
        "max_samples": effective_max_samples,
        "max_jobs": max_jobs,
        "max_jobs_alias_used": max_jobs_alias_used,
        "siesta_shell": siesta_shell,
        "samples_total": len(rows),
        "samples_ok": len([row for row in rows if row["status"] in {"ok", "staged", "skipped_existing"}]),
        "samples_failed": len(failed),
        "forbidden_reference_filenames": sorted(FORBIDDEN_REFERENCE_NAMES),
        "force_constants_used": False,
        "derivative_metrics_run": False,
        "rows": rows,
        "outputs": {
            "status_csv": str(output_reference_root / "derivative_siesta_reference_status.csv"),
            "manifest": str(output_reference_root / "derivative_siesta_reference_manifest.json"),
        },
    }
    write_csv(output_reference_root / "derivative_siesta_reference_status.csv", rows)
    write_json(output_reference_root / "derivative_siesta_reference_manifest.json", manifest)
    if failed and not diagnostic_only:
        raise DerivativeSiestaReferenceError(
            f"Derivative SIESTA reference stage failed for {len(failed)} sample(s). "
            f"See {output_reference_root / 'derivative_siesta_reference_manifest.json'}"
        )
    return manifest


def _effective_max_samples(*, max_samples: int | None, max_jobs: int | None) -> int | None:
    if max_samples is not None and max_jobs is not None and int(max_samples) != int(max_jobs):
        raise DerivativeSiestaReferenceError("--max-samples and deprecated --max-jobs disagree.")
    value = max_samples if max_samples is not None else max_jobs
    return int(value) if value is not None else None


def _command_display(command: str | list[str]) -> str:
    return " ".join(command) if isinstance(command, list) else command


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stencil-root", type=Path, required=True)
    parser.add_argument("--output-reference-root", type=Path, default=None)
    parser.add_argument("--source-dataset-root", type=Path, default=None)
    parser.add_argument("--existing-reference-root", type=Path, default=None)
    parser.add_argument("--siesta-command", default="siesta")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None, help="Deprecated alias for --max-samples.")
    parser.add_argument("--siesta-shell", action="store_true", help="Run --siesta-command through the shell. Default is argv execution.")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        manifest = run_derivative_siesta_references(
            stencil_root=args.stencil_root,
            output_reference_root=args.output_reference_root,
            source_dataset_root=args.source_dataset_root,
            existing_reference_root=args.existing_reference_root,
            siesta_command=args.siesta_command,
            overwrite=args.overwrite,
            skip_if_exists=args.skip_if_exists,
            diagnostic_only=args.diagnostic_only,
            workers=args.workers,
            max_samples=args.max_samples,
            max_jobs=args.max_jobs,
            max_jobs_alias_used=args.max_jobs is not None,
            siesta_shell=args.siesta_shell,
        )
    except DerivativeSiestaReferenceError as exc:
        print(f"[DERIVATIVE-SIESTA][ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"samples_total": manifest["samples_total"], "samples_ok": manifest["samples_ok"], "samples_failed": manifest["samples_failed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
