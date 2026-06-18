#!/usr/bin/env python3
"""Run or stage Hamiltonian predictions for derivative stencil structures."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_FILENAME = "ML_prediction.HSX"
STATUS_FIELDS = [
    "sample_id",
    "status",
    "model",
    "structure_dir",
    "prediction_dir",
    "prediction_path",
    "checkpoint",
    "model_dir",
    "command",
    "returncode",
    "started_at",
    "finished_at",
    "error",
]


class DerivativePredictionStageError(RuntimeError):
    """Raised when derivative prediction staging fails closed."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_structure_samples(stencil_root: Path) -> list[Path]:
    structures_root = stencil_root / "structures"
    if not structures_root.exists():
        raise DerivativePredictionStageError(f"Missing derivative stencil structures directory: {structures_root}")
    return sorted(path for path in structures_root.iterdir() if path.is_dir())


def existing_prediction(path: Path) -> Path | None:
    prediction = path / PREDICTION_FILENAME
    if prediction.exists() and prediction.is_file() and prediction.stat().st_size > 0:
        return prediction
    return None


def clean_prediction_dir(prediction_dir: Path, *, overwrite: bool) -> None:
    if overwrite and prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)


def copy_prediction(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / PREDICTION_FILENAME
    shutil.copy2(src, dst)
    return dst


def write_graph2mat_manifest(structures: list[Path], manifest_path: Path) -> None:
    rows = [
        {
            "sample_id": structure.name,
            "structure_path": str(structure / "RUN.fdf"),
            "sample_dir": str(structure),
        }
        for structure in structures
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "structure_path", "sample_dir"])
        writer.writeheader()
        writer.writerows(rows)


def graph2mat_command(
    *,
    python_executable: str,
    checkpoint: Path,
    manifest_path: Path,
    run_output_dir: Path,
    basis_files: str,
    accelerator: str,
    matrix_component_policy: str,
    n_matrix_components: int,
    loader_threads: int | None,
) -> list[str]:
    command = [
        python_executable,
        str(SCRIPT_DIR / "predict_model_on_dataset.py"),
        "--checkpoint",
        str(checkpoint),
        "--train-method",
        "md",
        "--test-set",
        "derivative_stencils",
        "--test-manifest",
        str(manifest_path),
        "--output-dir",
        str(run_output_dir),
        "--basis-files",
        basis_files,
        "--accelerator",
        accelerator,
        "--matrix-component-policy",
        matrix_component_policy,
        "--n-matrix-components",
        str(n_matrix_components),
        "--patch-graph2mat-basis-loading",
    ]
    if loader_threads is not None:
        command.extend(["--loader-threads", str(loader_threads)])
    return command


def run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": time.time(),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_deeph_command(
    command_template: str,
    *,
    stencil_root: Path,
    output_root: Path,
    model_dir: Path,
    use_shell: bool = False,
) -> dict[str, Any]:
    command = command_template.format(
        stencil_root=str(stencil_root),
        output_root=str(output_root),
        model_dir=str(model_dir),
    )
    command_args: str | list[str] = command if use_shell else shlex.split(command)
    started_at = time.time()
    completed = subprocess.run(
        command_args,
        cwd=REPO_ROOT,
        shell=use_shell,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command if use_shell else command_args,
        "shell": use_shell,
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": time.time(),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def status_row(
    *,
    sample_id: str,
    status: str,
    model: str,
    structure_dir: Path,
    prediction_dir: Path,
    checkpoint: Path | None = None,
    model_dir: Path | None = None,
    command: str | list[str] | None = None,
    returncode: int | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    prediction_path = prediction_dir / PREDICTION_FILENAME
    return {
        "sample_id": sample_id,
        "status": status,
        "model": model,
        "structure_dir": str(structure_dir),
        "prediction_dir": str(prediction_dir),
        "prediction_path": str(prediction_path) if prediction_path.exists() else "",
        "prediction_sha256": file_sha256(prediction_path),
        "checkpoint": str(checkpoint) if checkpoint else "",
        "model_dir": str(model_dir) if model_dir else "",
        "command": " ".join(command) if isinstance(command, list) else command or "",
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
    }


def stage_existing_predictions(
    *,
    structures: list[Path],
    model: str,
    output_root: Path,
    existing_prediction_root: Path,
    checkpoint: Path | None,
    model_dir: Path | None,
    overwrite: bool,
    skip_if_exists: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for structure in structures:
        sample_id = structure.name
        prediction_dir = output_root / sample_id
        started_at = time.time()
        if existing_prediction(prediction_dir) and skip_if_exists and not overwrite:
            rows.append(
                status_row(
                    sample_id=sample_id,
                    status="skipped_existing",
                    model=model,
                    structure_dir=structure,
                    prediction_dir=prediction_dir,
                    checkpoint=checkpoint,
                    model_dir=model_dir,
                    started_at=started_at,
                    finished_at=time.time(),
                )
            )
            continue
        clean_prediction_dir(prediction_dir, overwrite=overwrite)
        source_prediction = existing_prediction(existing_prediction_root / sample_id)
        if source_prediction is None:
            rows.append(
                status_row(
                    sample_id=sample_id,
                    status="error",
                    model=model,
                    structure_dir=structure,
                    prediction_dir=prediction_dir,
                    checkpoint=checkpoint,
                    model_dir=model_dir,
                    started_at=started_at,
                    finished_at=time.time(),
                    error="missing_existing_prediction",
                )
            )
            continue
        copy_prediction(source_prediction, prediction_dir)
        rows.append(
            status_row(
                sample_id=sample_id,
                status="staged",
                model=model,
                structure_dir=structure,
                prediction_dir=prediction_dir,
                checkpoint=checkpoint,
                model_dir=model_dir,
                started_at=started_at,
                finished_at=time.time(),
            )
        )
    return rows


def all_outputs_exist(structures: list[Path], output_root: Path) -> bool:
    return all(existing_prediction(output_root / structure.name) is not None for structure in structures)


def skipped_existing_rows(
    *,
    structures: list[Path],
    model: str,
    output_root: Path,
    checkpoint: Path | None,
    model_dir: Path | None,
) -> list[dict[str, Any]]:
    rows = []
    for structure in structures:
        now = time.time()
        rows.append(
            status_row(
                sample_id=structure.name,
                status="skipped_existing",
                model=model,
                structure_dir=structure,
                prediction_dir=output_root / structure.name,
                checkpoint=checkpoint,
                model_dir=model_dir,
                started_at=now,
                finished_at=now,
            )
        )
    return rows


def rows_from_output(
    *,
    structures: list[Path],
    model: str,
    output_root: Path,
    checkpoint: Path | None,
    model_dir: Path | None,
    command_record: dict[str, Any],
    overwrite: bool,
    skip_if_exists: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    command = command_record.get("command")
    for structure in structures:
        sample_id = structure.name
        prediction_dir = output_root / sample_id
        prediction = existing_prediction(prediction_dir)
        status = "predicted" if prediction is not None and int(command_record["returncode"]) == 0 else "error"
        error = "" if status == "predicted" else "missing_prediction" if int(command_record["returncode"]) == 0 else "prediction_command_failed"
        if prediction is not None and skip_if_exists and not overwrite and int(command_record["returncode"]) == 0:
            status = "skipped_existing"
            error = ""
        rows.append(
            status_row(
                sample_id=sample_id,
                status=status,
                model=model,
                structure_dir=structure,
                prediction_dir=prediction_dir,
                checkpoint=checkpoint,
                model_dir=model_dir,
                command=command,
                returncode=int(command_record["returncode"]),
                started_at=command_record.get("started_at"),
                finished_at=command_record.get("finished_at"),
                error=error,
            )
        )
    return rows


def run_derivative_predictions(
    *,
    stencil_root: Path,
    model: str,
    output_root: Path | None = None,
    checkpoint: Path | None = None,
    model_dir: Path | None = None,
    existing_prediction_root: Path | None = None,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    diagnostic_only: bool = False,
    max_samples: int | None = None,
    max_jobs: int | None = None,
    max_jobs_alias_used: bool = False,
    basis_files: str | None = None,
    accelerator: str = "cpu",
    matrix_component_policy: str = "h_only",
    n_matrix_components: int = 1,
    loader_threads: int | None = None,
    deeph_command: str | None = None,
    deeph_shell: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    model = str(model or "").strip().lower()
    if model not in {"graph2mat", "deeph"}:
        raise DerivativePredictionStageError("model must be one of: graph2mat, deeph.")
    output_root = output_root or stencil_root / "predicted_hamiltonians"
    structures = discover_structure_samples(stencil_root)
    effective_max_samples = _effective_max_samples(max_samples=max_samples, max_jobs=max_jobs)
    if effective_max_samples is not None:
        structures = structures[: max(0, int(effective_max_samples))]
    if not structures:
        raise DerivativePredictionStageError("No derivative stencil structures were found.")
    for structure in structures:
        if not (structure / "RUN.fdf").exists():
            raise DerivativePredictionStageError(f"Missing RUN.fdf for derivative stencil sample: {structure}")

    if existing_prediction_root is not None:
        rows = stage_existing_predictions(
            structures=structures,
            model=model,
            output_root=output_root,
            existing_prediction_root=existing_prediction_root,
            checkpoint=checkpoint,
            model_dir=model_dir,
            overwrite=overwrite,
            skip_if_exists=skip_if_exists,
        )
    elif skip_if_exists and not overwrite and all_outputs_exist(structures, output_root):
        rows = skipped_existing_rows(
            structures=structures,
            model=model,
            output_root=output_root,
            checkpoint=checkpoint,
            model_dir=model_dir,
        )
    elif model == "graph2mat":
        if checkpoint is None or not checkpoint.exists():
            raise DerivativePredictionStageError(f"Graph2Mat prediction requires an existing --checkpoint: {checkpoint}")
        if not basis_files:
            raise DerivativePredictionStageError("Graph2Mat prediction requires --basis-files when not staging existing predictions.")
        run_output_dir = output_root.parent / f".{model}_derivative_prediction_run"
        if overwrite and run_output_dir.exists():
            shutil.rmtree(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_output_dir / "derivative_prediction_manifest_input.csv"
        write_graph2mat_manifest(structures, manifest_path)
        command = graph2mat_command(
            python_executable=python_executable or sys.executable,
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            run_output_dir=run_output_dir,
            basis_files=basis_files,
            accelerator=accelerator,
            matrix_component_policy=matrix_component_policy,
            n_matrix_components=n_matrix_components,
            loader_threads=loader_threads,
        )
        command_record = run_command(command, cwd=REPO_ROOT)
        produced_root = run_output_dir / "predicted_hamiltonians"
        if produced_root.exists():
            if overwrite and output_root.exists():
                shutil.rmtree(output_root)
            output_root.mkdir(parents=True, exist_ok=True)
            for sample_dir in sorted(produced_root.iterdir()):
                if sample_dir.is_dir():
                    prediction = existing_prediction(sample_dir)
                    if prediction is not None:
                        copy_prediction(prediction, output_root / sample_dir.name)
        rows = rows_from_output(
            structures=structures,
            model=model,
            output_root=output_root,
            checkpoint=checkpoint,
            model_dir=model_dir,
            command_record=command_record,
            overwrite=overwrite,
            skip_if_exists=skip_if_exists,
        )
    else:
        if model_dir is None or not model_dir.exists():
            raise DerivativePredictionStageError(f"DeepH prediction requires an existing --model-dir: {model_dir}")
        if not deeph_command:
            raise DerivativePredictionStageError(
                "DeepH prediction requires --existing-prediction-root or --deeph-command. "
                "The command may use {stencil_root}, {output_root}, and {model_dir}."
            )
        command_record = run_deeph_command(
            deeph_command,
            stencil_root=stencil_root,
            output_root=output_root,
            model_dir=model_dir,
            use_shell=deeph_shell,
        )
        rows = rows_from_output(
            structures=structures,
            model=model,
            output_root=output_root,
            checkpoint=checkpoint,
            model_dir=model_dir,
            command_record=command_record,
            overwrite=overwrite,
            skip_if_exists=skip_if_exists,
        )

    failed = [row for row in rows if row["status"] == "error"]
    manifest = {
        "schema_version": "derivative_prediction_stage_v1",
        "stencil_root": str(stencil_root),
        "structures_root": str(stencil_root / "structures"),
        "output_root": str(output_root),
        "predicted_hamiltonians_root": str(output_root),
        "model": model,
        "checkpoint": str(checkpoint) if checkpoint else "",
        "model_dir": str(model_dir) if model_dir else "",
        "existing_prediction_root": str(existing_prediction_root) if existing_prediction_root else "",
        "overwrite": overwrite,
        "skip_if_exists": skip_if_exists,
        "diagnostic_only": diagnostic_only,
        "max_samples": effective_max_samples,
        "max_jobs": max_jobs,
        "max_jobs_alias_used": max_jobs_alias_used,
        "deeph_shell": deeph_shell,
        "python_executable": python_executable or sys.executable,
        "samples_total": len(rows),
        "samples_ok": len([row for row in rows if row["status"] in {"predicted", "staged", "skipped_existing"}]),
        "samples_failed": len(failed),
        "retraining_run": False,
        "output_filename": PREDICTION_FILENAME,
        "rows": rows,
        "outputs": {
            "status_csv": str(output_root / f"derivative_{model}_prediction_status.csv"),
            "manifest": str(output_root / f"derivative_{model}_prediction_manifest.json"),
        },
    }
    write_csv(output_root / f"derivative_{model}_prediction_status.csv", rows)
    write_json(output_root / f"derivative_{model}_prediction_manifest.json", manifest)
    if failed and not diagnostic_only:
        raise DerivativePredictionStageError(
            f"Derivative {model} prediction stage failed for {len(failed)} sample(s). "
            f"See {output_root / f'derivative_{model}_prediction_manifest.json'}"
        )
    return manifest


def _effective_max_samples(*, max_samples: int | None, max_jobs: int | None) -> int | None:
    if max_samples is not None and max_jobs is not None and int(max_samples) != int(max_jobs):
        raise DerivativePredictionStageError("--max-samples and deprecated --max-jobs disagree.")
    value = max_samples if max_samples is not None else max_jobs
    return int(value) if value is not None else None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stencil-root", type=Path, required=True)
    parser.add_argument("--model", choices=["graph2mat", "deeph"], required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--existing-prediction-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None, help="Deprecated alias for --max-samples.")
    parser.add_argument("--basis-files", default=None)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--matrix-component-policy", default="h_only")
    parser.add_argument("--n-matrix-components", type=int, default=1)
    parser.add_argument("--loader-threads", type=int, default=None)
    parser.add_argument("--deeph-command", default=None)
    parser.add_argument("--deeph-shell", action="store_true", help="Run --deeph-command through the shell. Default is argv execution.")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        manifest = run_derivative_predictions(
            stencil_root=args.stencil_root,
            model=args.model,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            model_dir=args.model_dir,
            existing_prediction_root=args.existing_prediction_root,
            overwrite=args.overwrite,
            skip_if_exists=args.skip_if_exists,
            diagnostic_only=args.diagnostic_only,
            max_samples=args.max_samples,
            max_jobs=args.max_jobs,
            max_jobs_alias_used=args.max_jobs is not None,
            basis_files=args.basis_files,
            accelerator=args.accelerator,
            matrix_component_policy=args.matrix_component_policy,
            n_matrix_components=args.n_matrix_components,
            loader_threads=args.loader_threads,
            deeph_command=args.deeph_command,
            deeph_shell=args.deeph_shell,
            python_executable=args.python_executable,
        )
    except DerivativePredictionStageError as exc:
        print(f"[DERIVATIVE-PREDICT][ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"samples_total": manifest["samples_total"], "samples_ok": manifest["samples_ok"], "samples_failed": manifest["samples_failed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
