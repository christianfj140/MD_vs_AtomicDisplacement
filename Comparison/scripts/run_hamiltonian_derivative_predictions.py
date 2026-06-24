#!/usr/bin/env python3
"""Run or stage Hamiltonian predictions for derivative stencil structures."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from deeph_config import (
    default_deeph_paths,
    render_inference_config,
    render_preprocess_config,
    validate_deeph_siesta_sample,
)
from deeph_prediction_adapter import (
    DeepHPredictionAdapterResult,
    adapt_deeph_prediction_sample,
    count_orbitals_from_orbital_types,
    parse_block_key,
    write_adapter_manifest,
)
from deeph_raw_global_equivalence_preflight import derive_deeph_to_siesta_basis_transform


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_FILENAME = "ML_prediction.HSX"
DEEPH_SPARSE_LAYOUT_KIND = "deeph_h5_reconstructed_siesta_sparse_layout_v1"
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


def expand_repo_tokens(text: str) -> str:
    return os.path.expandvars(text.replace("${REPO_ROOT}", str(REPO_ROOT)))


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


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
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


def read_ini(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if not config.read(path):
        raise DerivativePredictionStageError(f"Missing INI configuration file: {path}")
    return config


def discover_siesta_reference_samples(stencil_root: Path, *, structures: list[Path]) -> dict[str, Path]:
    reference_root = stencil_root / "siesta_hamiltonians"
    if not reference_root.exists():
        raise DerivativePredictionStageError(
            f"DeepH derivative prediction requires derivative SIESTA references under: {reference_root}"
        )
    references: dict[str, Path] = {}
    missing: list[str] = []
    for structure in structures:
        sample_id = structure.name
        sample_root = reference_root / sample_id
        if not sample_root.exists():
            missing.append(sample_id)
            continue
        references[sample_id] = sample_root
    if missing:
        raise DerivativePredictionStageError(
            "DeepH derivative prediction is missing SIESTA reference sample directories for: " + ", ".join(missing)
        )
    return references


def build_deeph_derivative_raw_mirror(*, references: dict[str, Path], raw_dir: Path) -> dict[str, Any]:
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sorted(references)):
        source_dir = references[sample_id]
        validate_deeph_siesta_sample(source_dir)
        raw_sample = raw_dir / f"{index:06d}_{sample_id}"
        raw_sample.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(source_dir.iterdir()):
            if not artifact.is_file() or artifact.name == PREDICTION_FILENAME:
                continue
            destination = raw_sample / artifact.name
            try:
                os.symlink(os.path.relpath(artifact, raw_sample), destination)
            except OSError:
                shutil.copy2(artifact, destination)
        rows.append({"sample_id": sample_id, "split": "test", "raw_dir": str(raw_sample), "source_dir": str(source_dir)})
    return {"raw_dir": str(raw_dir), "seed": 0, "rows": rows}


def deeph_command_uses_template(command_template: str | None) -> bool:
    template = str(command_template or "")
    return any(token in template for token in ("{stencil_root}", "{output_root}", "{model_dir}"))


def deeph_auto_backend_requested(command_template: str | None) -> bool:
    if command_template in (None, ""):
        return True
    return len(shlex.split(str(command_template))) == 1 and not deeph_command_uses_template(command_template)


def infer_deeph_cli(command_template: str | None, *, cli_name: str) -> str:
    if command_template not in (None, ""):
        parts = shlex.split(str(command_template))
        if len(parts) == 1:
            candidate = Path(parts[0])
            if candidate.name == "deeph-inference":
                sibling = candidate.with_name(cli_name)
                if sibling.exists():
                    return str(sibling)
            if cli_name == "deeph-inference":
                return parts[0]
    if cli_name == "deeph-inference":
        return "deeph-inference"
    import shutil as _shutil

    discovered = _shutil.which(cli_name)
    if discovered:
        return discovered
    inference_cli = infer_deeph_cli(command_template, cli_name="deeph-inference")
    inference_path = Path(inference_cli)
    sibling = inference_path.with_name(cli_name)
    if sibling.exists():
        return str(sibling)
    raise DerivativePredictionStageError(
        f"Could not resolve DeepH CLI '{cli_name}'. Provide derivative.deeph_command or add {cli_name} to PATH."
    )


def infer_deeph_source_repo(command_template: str | None) -> Path | None:
    try:
        inference_cli = Path(infer_deeph_cli(command_template, cli_name="deeph-inference")).expanduser()
    except DerivativePredictionStageError:
        return None
    if not inference_cli.is_absolute():
        resolved = shutil.which(str(inference_cli))
        inference_cli = Path(resolved).expanduser() if resolved else inference_cli
    for parent in inference_cli.resolve(strict=False).parents:
        if parent.joinpath("deeph").is_dir():
            return parent
    return None


def deeph_command_env(command_template: str | None) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    inference_cli = Path(infer_deeph_cli(command_template, cli_name="deeph-inference")).expanduser()
    if inference_cli.is_absolute():
        bin_dir = str(inference_cli.resolve(strict=False).parent)
        current_path = env.get("PATH", "")
        env["PATH"] = bin_dir if not current_path else f"{bin_dir}{os.pathsep}{current_path}"
    source_repo = infer_deeph_source_repo(command_template)
    if source_repo is not None:
        current_pythonpath = env.get("PYTHONPATH", "")
        repo_path = str(source_repo)
        env["PYTHONPATH"] = repo_path if not current_pythonpath else f"{repo_path}{os.pathsep}{current_pythonpath}"
    return env


def deeph_command_cwd(command_template: str | None) -> Path:
    return infer_deeph_source_repo(command_template) or REPO_ROOT


def deeph_runtime_settings(model_dir: Path, *, python_executable: str, command_template: str | None) -> dict[str, Any]:
    config_path = model_dir / "config.ini"
    radius = -1.0
    disable_cuda = True
    device = "cpu"
    huge_structure = False
    if config_path.exists():
        config = read_ini(config_path)
        radius = config.getfloat("graph", "radius", fallback=-1.0)
        disable_cuda = config.getboolean("basic", "disable_cuda", fallback=True)
        device = config.get("basic", "device", fallback="cpu")
    python_path = python_executable
    if python_path in ("", sys.executable):
        inference_cli = Path(infer_deeph_cli(command_template, cli_name="deeph-inference"))
        sibling_python = inference_cli.with_name("python")
        sibling_python3 = inference_cli.with_name("python3")
        if sibling_python.exists():
            python_path = str(sibling_python)
        elif sibling_python3.exists():
            python_path = str(sibling_python3)
    return {
        "radius": radius,
        "disable_cuda": disable_cuda,
        "device": device,
        "huge_structure": huge_structure,
        "python_interpreter": python_path,
    }


def first_matching_file(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        raise DerivativePredictionStageError(f"DeepH derivative prediction requires *{suffix} under {directory}")
    return matches[0]


def reconstruct_deeph_sparse_layout_prediction(
    *,
    prediction_h5: Path,
    processed_sample_dir: Path,
    siesta_reference_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    import h5py
    import numpy as np
    from scipy import sparse

    orbital_types = processed_sample_dir / "orbital_types.dat"
    orbital_counts = count_orbitals_from_orbital_types(orbital_types)
    offsets = np.cumsum([0, *orbital_counts])
    unit_orbitals = int(offsets[-1])
    r_list = [
        tuple(int(value) for value in line.split())
        for line in (processed_sample_dir / "R_list.dat").read_text(encoding="utf-8").splitlines()
        if line.split()
    ]
    if not r_list:
        raise DerivativePredictionStageError(f"DeepH processed sample has no R_list.dat entries: {processed_sample_dir}")
    r_index = {vector: idx for idx, vector in enumerate(r_list)}
    orb_indx = first_matching_file(siesta_reference_dir, ".ORB_INDX")
    transform = derive_deeph_to_siesta_basis_transform(
        row={"artifact_paths": {"orb_indx": str(orb_indx)}},
        manifest_dir=REPO_ROOT,
        orbital_types=orbital_types,
        n_orbitals=unit_orbitals,
    )
    permutation = np.asarray(transform.get("permutation") or [], dtype=int)
    signs = np.asarray(transform.get("signs") or [], dtype=float)
    matrix = sparse.lil_matrix((unit_orbitals, unit_orbitals * len(r_list)), dtype=np.complex128)
    with h5py.File(prediction_h5, "r") as handle:
        for key in handle.keys():
            lattice_r, atom_i, atom_j = parse_block_key(key)
            column_block = r_index.get(tuple(lattice_r))
            if column_block is None:
                raise DerivativePredictionStageError(
                    f"DeepH prediction block {key} uses R={tuple(lattice_r)} not present in {processed_sample_dir / 'R_list.dat'}"
                )
            row_slice = slice(int(offsets[atom_i]), int(offsets[atom_i + 1]))
            col_slice = slice(int(offsets[atom_j]), int(offsets[atom_j + 1]))
            block = np.asarray(handle[key][()])
            row_perm = permutation[row_slice] - int(offsets[atom_i])
            col_perm = permutation[col_slice] - int(offsets[atom_j])
            row_sign = signs[row_slice]
            col_sign = signs[col_slice]
            block = block[np.ix_(row_perm, col_perm)]
            block = (row_sign[:, None] * block) * col_sign[None, :]
            r0, r1 = int(offsets[atom_i]), int(offsets[atom_i + 1])
            c0 = column_block * unit_orbitals + int(offsets[atom_j])
            c1 = column_block * unit_orbitals + int(offsets[atom_j + 1])
            matrix[r0:r1, c0:c1] = block
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        sparse.save_npz(handle, matrix.tocsr())
    reconstructed = matrix.tocsr()
    return {
        "kind": DEEPH_SPARSE_LAYOUT_KIND,
        "nnz": int(reconstructed.nnz),
        "shape_rows": int(reconstructed.shape[0]),
        "shape_cols": int(reconstructed.shape[1]),
    }


def run_deeph_auto_backend(
    *,
    stencil_root: Path,
    structures: list[Path],
    output_root: Path,
    model_dir: Path,
    overwrite: bool,
    skip_if_exists: bool,
    diagnostic_only: bool,
    command_template: str | None,
    python_executable: str,
) -> dict[str, Any]:
    references = discover_siesta_reference_samples(stencil_root, structures=structures)
    deeph_paths = default_deeph_paths(output_root.parent)
    if overwrite and deeph_paths.root.exists():
        shutil.rmtree(deeph_paths.root)
    settings = deeph_runtime_settings(model_dir, python_executable=python_executable, command_template=command_template)
    raw_mirror = build_deeph_derivative_raw_mirror(references=references, raw_dir=deeph_paths.raw_dir)
    render_preprocess_config(
        deeph_paths.preprocess_config,
        raw_dir=deeph_paths.raw_dir,
        processed_dir=deeph_paths.processed_dir,
        multiprocessing=0,
        local_coordinate=True,
        get_s=True,
        radius=float(settings["radius"]),
        julia_interpreter="",
    )
    preprocess_cli = infer_deeph_cli(command_template, cli_name="deeph-preprocess")
    command_cwd = deeph_command_cwd(command_template)
    command_env = deeph_command_env(command_template)
    preprocess_record = run_command(
        [preprocess_cli, "--config", str(deeph_paths.preprocess_config)],
        cwd=command_cwd,
        env=command_env,
    )
    rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    adapter_results: list[DeepHPredictionAdapterResult] = []
    sample_index = {row["sample_id"]: row for row in raw_mirror["rows"]}
    if int(preprocess_record["returncode"]) != 0:
        for structure in structures:
            rows.append(
                status_row(
                    sample_id=structure.name,
                    status="error",
                    model="deeph",
                    structure_dir=structure,
                    prediction_dir=output_root / structure.name,
                    model_dir=model_dir,
                    command=preprocess_record["command"],
                    returncode=int(preprocess_record["returncode"]),
                    started_at=preprocess_record["started_at"],
                    finished_at=preprocess_record["finished_at"],
                    error="deeph_preprocess_failed",
                )
            )
        return {
            "rows": rows,
            "adapter_results": [],
            "layout_rows": [],
            "runtime": {
                "mode": "deeph_auto_backend",
                "preprocess_command": " ".join(preprocess_record["command"]),
                "preprocess_returncode": int(preprocess_record["returncode"]),
                "preprocess_stdout": preprocess_record["stdout"],
                "preprocess_stderr": preprocess_record["stderr"],
                "cwd": str(command_cwd),
                "pythonpath_prefix": str(infer_deeph_source_repo(command_template) or ""),
            },
        }

    inference_cli = infer_deeph_cli(command_template, cli_name="deeph-inference")
    for structure in structures:
        sample = sample_index[structure.name]
        raw_sample_dir = Path(str(sample["raw_dir"]))
        processed_sample_dir = deeph_paths.processed_dir / raw_sample_dir.name
        if not processed_sample_dir.exists():
            rows.append(
                status_row(
                    sample_id=structure.name,
                    status="error",
                    model="deeph",
                    structure_dir=structure,
                    prediction_dir=output_root / structure.name,
                    model_dir=model_dir,
                    command=[preprocess_cli, "--config", str(deeph_paths.preprocess_config)],
                    returncode=0,
                    error="deeph_processed_sample_missing",
                )
            )
            continue
        work_dir = deeph_paths.inference_dir / raw_sample_dir.name
        if overwrite and work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(processed_sample_dir.iterdir()):
            if item.is_file():
                destination = work_dir / item.name
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                try:
                    os.symlink(os.path.relpath(item, work_dir), destination)
                except OSError:
                    shutil.copy2(item, destination)
        inference_config = deeph_paths.config_dir / "inference" / f"{raw_sample_dir.name}.ini"
        render_inference_config(
            inference_config,
            work_dir=work_dir,
            trained_model_dir=model_dir,
            python_interpreter=str(settings["python_interpreter"]),
            interface="openmx",
            task=[3, 4],
            disable_cuda=bool(settings["disable_cuda"]),
            device=str(settings["device"]),
            huge_structure=bool(settings["huge_structure"]),
            restore_blocks_py=True,
            radius=float(settings["radius"]),
        )
        command = [inference_cli, "--config", str(inference_config)]
        command_record = run_command(command, cwd=command_cwd, env=command_env)
        prediction_dir = output_root / structure.name
        clean_prediction_dir(prediction_dir, overwrite=overwrite)
        h5_prediction_path = work_dir / "hamiltonians_pred.h5"
        rh_prediction_path = work_dir / "rh_pred.h5"
        if h5_prediction_path.exists():
            shutil.copy2(h5_prediction_path, prediction_dir / h5_prediction_path.name)
        if rh_prediction_path.exists():
            shutil.copy2(rh_prediction_path, prediction_dir / rh_prediction_path.name)
        status = "error"
        error = "prediction_command_failed" if int(command_record["returncode"]) != 0 else "missing_prediction"
        adapter_status = ""
        metrics_ready = False
        reconstructed_layout: dict[str, Any] | None = None
        if int(command_record["returncode"]) == 0 and h5_prediction_path.exists():
            try:
                adapter_result = adapt_deeph_prediction_sample(
                    work_dir=work_dir,
                    processed_sample_dir=processed_sample_dir,
                    sample_id=structure.name,
                )
                adapter_results.append(adapter_result)
                adapter_status = adapter_result.status
                metrics_ready = bool(adapter_result.metrics_ready)
                write_json(prediction_dir / "deeph_adapter_result.json", adapter_result.to_dict())
                reconstructed_layout = reconstruct_deeph_sparse_layout_prediction(
                    prediction_h5=h5_prediction_path,
                    processed_sample_dir=processed_sample_dir,
                    siesta_reference_dir=references[structure.name],
                    output_path=prediction_dir / PREDICTION_FILENAME,
                )
                layout_rows.append({"sample_id": structure.name, **reconstructed_layout})
                status = "predicted"
                error = ""
            except Exception as exc:
                error = f"deeph_adapter_failed:{type(exc).__name__}"
        row = status_row(
            sample_id=structure.name,
            status=status,
            model="deeph",
            structure_dir=structure,
            prediction_dir=prediction_dir,
            model_dir=model_dir,
            command=command_record["command"],
            returncode=int(command_record["returncode"]),
            started_at=command_record["started_at"],
            finished_at=command_record["finished_at"],
            error=error,
        )
        row["prediction_h5_path"] = str(prediction_dir / h5_prediction_path.name) if h5_prediction_path.exists() else ""
        row["adapter_status"] = adapter_status
        row["metrics_ready"] = metrics_ready
        if reconstructed_layout is not None:
            row["prediction_layout"] = reconstructed_layout["kind"]
            row["prediction_shape"] = [reconstructed_layout["shape_rows"], reconstructed_layout["shape_cols"]]
        rows.append(row)

    adapter_manifest_path = deeph_paths.inference_dir / "adapter_manifest.json"
    adapter_manifest = write_adapter_manifest(adapter_manifest_path, adapter_results) if adapter_results else {}
    layout_note_path = output_root / "deeph_sparse_layout_note.json"
    layout_note = {
        "kind": DEEPH_SPARSE_LAYOUT_KIND,
        "description": (
            "ML_prediction.HSX stores a scipy sparse NPZ payload in the SIESTA sparse row/column layout "
            "reconstructed from DeepH hamiltonians_pred.h5 and the sample HSX sparsity pattern. "
            "This is diagnostic-only and does not prove raw/global HSX equivalence."
        ),
        "rows": layout_rows,
    }
    if layout_rows:
        write_json(layout_note_path, layout_note)
    return {
        "rows": rows,
        "adapter_results": [result.to_dict() for result in adapter_results],
        "adapter_manifest": str(adapter_manifest_path) if adapter_results else "",
        "adapter_manifest_payload": adapter_manifest,
        "layout_note": str(layout_note_path) if layout_rows else "",
        "layout_rows": layout_rows,
        "runtime": {
            "mode": "deeph_auto_backend",
            "preprocess_command": " ".join(preprocess_record["command"]),
            "preprocess_returncode": int(preprocess_record["returncode"]),
            "preprocess_stdout": preprocess_record["stdout"],
            "preprocess_stderr": preprocess_record["stderr"],
            "inference_cli": inference_cli,
            "python_interpreter": str(settings["python_interpreter"]),
            "cwd": str(command_cwd),
            "pythonpath_prefix": str(infer_deeph_source_repo(command_template) or ""),
        },
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
    stencil_root = stencil_root.expanduser().resolve(strict=False)
    output_root = output_root or stencil_root / "predicted_hamiltonians"
    output_root = output_root.expanduser().resolve(strict=False)
    checkpoint = checkpoint.expanduser().resolve(strict=False) if checkpoint is not None else None
    model_dir = model_dir.expanduser().resolve(strict=False) if model_dir is not None else None
    existing_prediction_root = (
        existing_prediction_root.expanduser().resolve(strict=False) if existing_prediction_root is not None else None
    )
    if basis_files:
        basis_files = expand_repo_tokens(str(basis_files))
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
            raise DerivativePredictionStageError(
                "Graph2Mat prediction requires --basis-files when not staging existing predictions. "
                "For workflow payloads, set derivative.basis_files to the required Graph2Mat basis XML glob or file list."
            )
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
        if deeph_auto_backend_requested(deeph_command):
            extras = run_deeph_auto_backend(
                stencil_root=stencil_root,
                structures=structures,
                output_root=output_root,
                model_dir=model_dir,
                overwrite=overwrite,
                skip_if_exists=skip_if_exists,
                diagnostic_only=diagnostic_only,
                command_template=deeph_command,
                python_executable=python_executable or sys.executable,
            )
            rows = extras["rows"]
        elif not deeph_command:
            raise DerivativePredictionStageError(
                "DeepH prediction requires --existing-prediction-root or --deeph-command. "
                "The command may use {stencil_root}, {output_root}, and {model_dir}."
            )
        else:
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
        "basis_files": basis_files or "",
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
    if model == "deeph" and "extras" in locals():
        manifest.update(
            {
                "runtime": extras.get("runtime", {}),
                "adapter_manifest": extras.get("adapter_manifest", ""),
                "layout_note": extras.get("layout_note", ""),
            }
        )
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
