#!/usr/bin/env python3
"""Direct DeepH dH_pred/dR predictions from hamiltonians_grad_pred.h5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deeph_config import default_deeph_paths, render_inference_config, render_preprocess_config  # noqa: E402
from deeph_prediction_adapter import adapt_deeph_prediction_sample  # noqa: E402
from hamiltonian_derivative_stencil import (  # noqa: E402
    DEEPH_PREDICTION_METHOD_AUTOGRAD,
    EXPECTED_DERIVATIVE_UNITS,
    EXPECTED_DISPLACEMENT_UNITS,
    EXPECTED_HAMILTONIAN_UNITS,
    PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
    REFERENCE_DERIVATIVE_METHOD_SIESTA,
    VALID_AXES,
    direct_derivative_prediction_basename,
    sparse_hermiticity_defect,
)
from run_hamiltonian_derivative_predictions import (  # noqa: E402
    DerivativePredictionStageError,
    build_deeph_derivative_raw_mirror,
    deeph_command_cwd,
    deeph_command_env,
    deeph_runtime_settings,
    discover_siesta_reference_samples,
    file_sha256,
    infer_deeph_cli,
    infer_deeph_source_repo,
    reconstruct_deeph_sparse_layout_prediction,
    run_command,
)


MANIFEST_FILENAME = "derivative_deeph_autograd_prediction_manifest.json"
STATUS_FILENAME = "derivative_deeph_autograd_prediction_status.csv"
STATUS_FIELDS = [
    "base_structure_sample_id",
    "base_sample_id",
    "atom_index_zero_based",
    "axis",
    "axis_index",
    "status",
    "prediction_path",
    "metadata_path",
    "nnz",
    "shape_rows",
    "shape_cols",
    "reference_delta_ang_values",
    "error",
]


class DeepHAutogradDerivativePredictionError(RuntimeError):
    """Raised when direct DeepH derivative prediction fails closed."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def read_structure_metadata(structures_root: Path) -> list[dict[str, Any]]:
    if not structures_root.exists():
        raise DeepHAutogradDerivativePredictionError(
            f"Missing derivative stencil structures directory: {structures_root}"
        )
    payloads: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in structures_root.iterdir() if path.is_dir()):
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DeepHAutogradDerivativePredictionError(
                f"Unreadable stencil metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise DeepHAutogradDerivativePredictionError(
                f"Stencil metadata must be a JSON object: {metadata_path}"
            )
        payload = dict(payload)
        payload["_sample_dir"] = sample_dir
        payload["_structure_sample_id"] = sample_dir.name
        payloads.append(payload)
    if not payloads:
        raise DeepHAutogradDerivativePredictionError(
            f"No stencil sample metadata found under {structures_root}"
        )
    return payloads


def _is_base_sample(metadata: dict[str, Any]) -> bool:
    if bool(metadata.get("is_reference")):
        return True
    sign = metadata.get("sign")
    return isinstance(sign, (int, float)) and int(sign) == 0


def collect_derivative_requests(
    payloads: list[dict[str, Any]],
    *,
    base_sample_ids: list[str],
    atoms: list[int],
    axes: list[str],
    max_base_structures: int | None,
) -> list[dict[str, Any]]:
    base_samples: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if _is_base_sample(payload):
            base_id = str(payload.get("base_sample_id") or payload.get("sample_id") or "").strip()
            if base_id:
                base_samples[base_id] = payload

    requests: dict[str, dict[str, Any]] = {}
    missing_bases: set[str] = set()
    for payload in payloads:
        if _is_base_sample(payload):
            continue
        base_id = str(payload.get("base_sample_id") or "").strip()
        atom_index = payload.get("atom_index_zero_based")
        axis = str(payload.get("axis") or "").strip().lower()
        delta_ang = payload.get("delta_ang", payload.get("amplitude_ang"))
        if not base_id or atom_index is None or axis not in VALID_AXES:
            continue
        if base_sample_ids and base_id not in base_sample_ids:
            continue
        if atoms and int(atom_index) not in atoms:
            continue
        if axes and axis not in axes:
            continue
        base_payload = base_samples.get(base_id)
        if base_payload is None:
            missing_bases.add(base_id)
            continue
        request = requests.setdefault(
            base_id,
            {
                "base_sample_id": base_id,
                "base_structure_sample_id": base_payload["_structure_sample_id"],
                "base_sample_dir": base_payload["_sample_dir"],
                "split": base_payload.get("split"),
                "pairs": {},
            },
        )
        deltas = request["pairs"].setdefault((int(atom_index), axis), set())
        try:
            delta_value = float(delta_ang)
        except (TypeError, ValueError):
            delta_value = None
        if delta_value is not None and delta_value > 0:
            deltas.add(delta_value)

    if missing_bases:
        raise DeepHAutogradDerivativePredictionError(
            "DeepH autograd derivatives require base stencil structures, missing for: "
            + ", ".join(sorted(missing_bases))
            + ". Rebuild stencils with --include-base."
        )
    if not requests:
        raise DeepHAutogradDerivativePredictionError(
            "No displaced stencil samples matched the requested base/atom/axis filters."
        )
    ordered = [requests[key] for key in sorted(requests)]
    if max_base_structures is not None:
        ordered = ordered[: max(0, int(max_base_structures))]
    if not ordered:
        raise DeepHAutogradDerivativePredictionError("No base structures selected.")
    return ordered


def _request_pairs(request: dict[str, Any]) -> list[tuple[int, str, list[float]]]:
    return [
        (int(atom_index), str(axis), sorted(float(delta) for delta in deltas))
        for (atom_index, axis), deltas in sorted(request["pairs"].items())
    ]


def _row_for_pair(
    *,
    request: dict[str, Any],
    atom_index: int,
    axis: str,
    reference_deltas: list[float],
    status: str = "error",
    error: str = "",
) -> dict[str, Any]:
    axis_index = VALID_AXES[axis]
    return {
        "base_structure_sample_id": request["base_structure_sample_id"],
        "base_sample_id": request["base_sample_id"],
        "atom_index_zero_based": atom_index,
        "axis": axis,
        "axis_index": axis_index,
        "status": status,
        "prediction_path": "",
        "metadata_path": "",
        "nnz": None,
        "shape_rows": None,
        "shape_cols": None,
        "reference_delta_ang_values": ";".join(f"{delta:g}" for delta in reference_deltas),
        "error": error,
    }


def _select_gradient_block(block: Any, atom_index: int, axis_index: int) -> Any:
    if getattr(block, "ndim", 0) != 4:
        raise DeepHAutogradDerivativePredictionError(
            f"DeepH grad block must have shape (rows, cols, atoms, axes), got {getattr(block, 'shape', None)}"
        )
    if atom_index < 0 or atom_index >= int(block.shape[2]):
        raise DeepHAutogradDerivativePredictionError(
            f"Requested atom index {atom_index} outside DeepH grad block atom axis {block.shape[2]}."
        )
    if axis_index < 0 or axis_index >= int(block.shape[3]):
        raise DeepHAutogradDerivativePredictionError(
            f"Requested axis index {axis_index} outside DeepH grad block axis {block.shape[3]}."
        )
    return block[..., atom_index, axis_index]


def _prediction_file_metadata(
    *,
    request: dict[str, Any],
    atom_index: int,
    axis: str,
    reference_deltas: list[float],
    matrix: sparse.csr_matrix,
    model_dir: Path,
    h5_path: Path,
    layout: dict[str, Any],
    adapter_fields: dict[str, Any],
) -> dict[str, Any]:
    axis_index = VALID_AXES[axis]
    hermiticity = sparse_hermiticity_defect(matrix)
    payload = {
        "schema": "deeph_autograd_direct_derivative_v1",
        "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
        "predicted_derivative_method": PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
        "reference_delta_ang": reference_deltas[0] if len(reference_deltas) == 1 else None,
        "reference_delta_ang_values": reference_deltas,
        "predicted_delta_ang": None,
        "atom_index": atom_index + 1,
        "atom_index_zero_based": atom_index,
        "atom_index_one_based": atom_index + 1,
        "axis_index": axis_index,
        "axis_name": axis,
        "axis": axis,
        "units": "eV/Angstrom",
        "hamiltonian_units": EXPECTED_HAMILTONIAN_UNITS,
        "displacement_units": EXPECTED_DISPLACEMENT_UNITS,
        "derivative_units": EXPECTED_DERIVATIVE_UNITS,
        "deeph_prediction_method": DEEPH_PREDICTION_METHOD_AUTOGRAD,
        "topology_fixed": True,
        "with_grad": True,
        "base_sample_id": request["base_sample_id"],
        "base_structure_sample_id": request["base_structure_sample_id"],
        "split": request.get("split"),
        "model_dir": str(model_dir),
        "source_prediction_h5": str(h5_path),
        "source_prediction_h5_sha256": file_sha256(h5_path),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "dh_hermiticity_defect": None if not math.isfinite(hermiticity) else hermiticity,
        "layout": layout,
    }
    payload.update(adapter_fields)
    return payload


def _link_processed_sample(processed_sample_dir: Path, work_dir: Path, *, overwrite: bool) -> None:
    if overwrite and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(processed_sample_dir.iterdir()):
        if not item.is_file():
            continue
        destination = work_dir / item.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        try:
            os.symlink(os.path.relpath(item, work_dir), destination)
        except OSError:
            shutil.copy2(item, destination)


def _mark_skipped_existing(row: dict[str, Any], npz_path: Path, json_path: Path) -> dict[str, Any]:
    matrix = sparse.load_npz(npz_path).tocsr()
    row.update(
        {
            "status": "skipped_existing",
            "prediction_path": str(npz_path),
            "metadata_path": str(json_path),
            "nnz": int(matrix.nnz),
            "shape_rows": int(matrix.shape[0]),
            "shape_cols": int(matrix.shape[1]),
            "error": "",
        }
    )
    return row


def _write_outputs(output_root: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    write_csv(output_root / STATUS_FILENAME, rows)
    write_json(output_root / MANIFEST_FILENAME, manifest)


def run_deeph_autograd_derivative_predictions(args: argparse.Namespace) -> dict[str, Any]:
    stencil_root = args.stencil_root.expanduser().resolve(strict=False)
    output_root = (
        args.output_root.expanduser().resolve(strict=False)
        if args.output_root is not None
        else stencil_root / "predicted_derivative_hamiltonians"
    )
    model_dir = args.model_dir.expanduser().resolve(strict=False)
    if not model_dir.exists():
        raise DeepHAutogradDerivativePredictionError(
            f"DeepH autograd derivative prediction requires an existing --model-dir: {model_dir}"
        )

    atoms = [int(atom) for atom in (args.atoms or [])]
    axes = [str(axis).strip().lower() for axis in (args.axes or [])]
    invalid_axes = [axis for axis in axes if axis not in VALID_AXES]
    if invalid_axes:
        raise DeepHAutogradDerivativePredictionError(
            f"Unsupported axes: {', '.join(invalid_axes)}. Use x, y, z."
        )

    max_base_structures = args.max_base_structures
    if args.max_samples is not None:
        if max_base_structures is not None and int(max_base_structures) != int(args.max_samples):
            raise DeepHAutogradDerivativePredictionError("--max-samples and --max-base-structures disagree.")
        max_base_structures = int(args.max_samples)

    payloads = read_structure_metadata(stencil_root / "structures")
    requests = collect_derivative_requests(
        payloads,
        base_sample_ids=[str(item) for item in (args.base_sample_id or [])],
        atoms=atoms,
        axes=axes,
        max_base_structures=max_base_structures,
    )

    rows: list[dict[str, Any]] = []
    started_at = time.time()
    if args.skip_if_exists and not args.overwrite:
        all_existing = True
        for request in requests:
            structure_id = request["base_structure_sample_id"]
            for atom_index, axis, reference_deltas in _request_pairs(request):
                basename = direct_derivative_prediction_basename(atom_index, VALID_AXES[axis])
                npz_path = output_root / structure_id / f"{basename}.npz"
                json_path = output_root / structure_id / f"{basename}.json"
                if npz_path.exists():
                    rows.append(
                        _mark_skipped_existing(
                            _row_for_pair(
                                request=request,
                                atom_index=atom_index,
                                axis=axis,
                                reference_deltas=reference_deltas,
                            ),
                            npz_path,
                            json_path,
                        )
                    )
                else:
                    all_existing = False
        if all_existing and rows:
            manifest = _manifest_payload(args, stencil_root, output_root, model_dir, requests, rows, started_at)
            _write_outputs(output_root, rows, manifest)
            return manifest
        rows = []

    structures = [Path(request["base_sample_dir"]) for request in requests]
    references = discover_siesta_reference_samples(stencil_root, structures=structures)
    deeph_paths = default_deeph_paths(output_root.parent)
    if args.overwrite and deeph_paths.root.exists():
        shutil.rmtree(deeph_paths.root)

    settings = deeph_runtime_settings(
        model_dir,
        python_executable=args.python_executable or sys.executable,
        command_template=args.deeph_command,
    )
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
    preprocess_cli = infer_deeph_cli(args.deeph_command, cli_name="deeph-preprocess")
    command_cwd = deeph_command_cwd(args.deeph_command)
    command_env = deeph_command_env(args.deeph_command)
    preprocess_record = run_command([preprocess_cli, "--config", str(deeph_paths.preprocess_config)], cwd=command_cwd, env=command_env)
    if int(preprocess_record["returncode"]) != 0:
        for request in requests:
            for atom_index, axis, reference_deltas in _request_pairs(request):
                rows.append(
                    _row_for_pair(
                        request=request,
                        atom_index=atom_index,
                        axis=axis,
                        reference_deltas=reference_deltas,
                        error="deeph_preprocess_failed",
                    )
                )
        manifest = _manifest_payload(args, stencil_root, output_root, model_dir, requests, rows, started_at)
        manifest["runtime"] = {
            "mode": "deeph_autograd_backend",
            "preprocess_command": " ".join(preprocess_record["command"]),
            "preprocess_returncode": int(preprocess_record["returncode"]),
            "preprocess_stdout": preprocess_record["stdout"],
            "preprocess_stderr": preprocess_record["stderr"],
            "cwd": str(command_cwd),
            "pythonpath_prefix": str(infer_deeph_source_repo(args.deeph_command) or ""),
        }
        _write_outputs(output_root, rows, manifest)
        raise DeepHAutogradDerivativePredictionError(
            f"DeepH preprocess failed. See {output_root / MANIFEST_FILENAME}"
        )

    sample_index = {row["sample_id"]: row for row in raw_mirror["rows"]}
    inference_cli = infer_deeph_cli(args.deeph_command, cli_name="deeph-inference")
    inference_records: list[dict[str, Any]] = []

    for request in requests:
        structure_id = request["base_structure_sample_id"]
        sample = sample_index[structure_id]
        raw_sample_dir = Path(str(sample["raw_dir"]))
        processed_sample_dir = deeph_paths.processed_dir / raw_sample_dir.name
        if not processed_sample_dir.exists():
            for atom_index, axis, reference_deltas in _request_pairs(request):
                rows.append(
                    _row_for_pair(
                        request=request,
                        atom_index=atom_index,
                        axis=axis,
                        reference_deltas=reference_deltas,
                        error="deeph_processed_sample_missing",
                    )
                )
            continue

        work_dir = deeph_paths.inference_dir / raw_sample_dir.name
        _link_processed_sample(processed_sample_dir, work_dir, overwrite=args.overwrite)
        pairs = _request_pairs(request)
        inference_config = deeph_paths.config_dir / "inference_autograd" / f"{raw_sample_dir.name}.ini"
        render_inference_config(
            inference_config,
            work_dir=work_dir,
            trained_model_dir=model_dir,
            python_interpreter=str(settings["python_interpreter"]),
            interface="openmx",
            task=[3],
            disable_cuda=bool(settings["disable_cuda"]),
            device=str(settings["device"]),
            huge_structure=bool(settings["huge_structure"]),
            restore_blocks_py=True,
            radius=float(settings["radius"]),
            with_grad=True,
            grad_atom_indices=sorted({atom for atom, _axis, _deltas in pairs}),
            grad_axis_indices=sorted({VALID_AXES[axis] for _atom, axis, _deltas in pairs}),
        )
        command_record = run_command([inference_cli, "--config", str(inference_config)], cwd=command_cwd, env=command_env)
        inference_records.append(
            {
                "base_structure_sample_id": structure_id,
                "command": command_record["command"],
                "returncode": int(command_record["returncode"]),
                "started_at": command_record["started_at"],
                "finished_at": command_record["finished_at"],
            }
        )
        grad_h5 = work_dir / "hamiltonians_grad_pred.h5"
        pred_h5 = work_dir / "hamiltonians_pred.h5"
        sample_output_dir = output_root / structure_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        if grad_h5.exists():
            shutil.copy2(grad_h5, sample_output_dir / grad_h5.name)
        if pred_h5.exists():
            shutil.copy2(pred_h5, sample_output_dir / pred_h5.name)
        adapter_fields = _deeph_base_equivalence_fields(
            work_dir=work_dir,
            processed_sample_dir=processed_sample_dir,
            sample_id=structure_id,
            output_dir=sample_output_dir,
        )

        for atom_index, axis, reference_deltas in pairs:
            axis_index = VALID_AXES[axis]
            basename = direct_derivative_prediction_basename(atom_index, axis_index)
            npz_path = sample_output_dir / f"{basename}.npz"
            json_path = sample_output_dir / f"{basename}.json"
            row = _row_for_pair(
                request=request,
                atom_index=atom_index,
                axis=axis,
                reference_deltas=reference_deltas,
            )
            try:
                if npz_path.exists() and args.skip_if_exists and not args.overwrite:
                    rows.append(_mark_skipped_existing(row, npz_path, json_path))
                    continue
                if int(command_record["returncode"]) != 0:
                    raise DeepHAutogradDerivativePredictionError("prediction_command_failed")
                if not grad_h5.exists():
                    raise DeepHAutogradDerivativePredictionError("missing_hamiltonians_grad_pred_h5")
                layout = reconstruct_deeph_sparse_layout_prediction(
                    prediction_h5=grad_h5,
                    processed_sample_dir=processed_sample_dir,
                    siesta_reference_dir=references[structure_id],
                    output_path=npz_path,
                    block_transform=lambda block, atom=atom_index, axis_i=axis_index: _select_gradient_block(
                        block, atom, axis_i
                    ),
                )
                matrix = sparse.load_npz(npz_path).tocsr().astype("float64")
                with npz_path.open("wb") as handle:
                    sparse.save_npz(handle, matrix)
                write_json(
                    json_path,
                    _prediction_file_metadata(
                        request=request,
                        atom_index=atom_index,
                        axis=axis,
                        reference_deltas=reference_deltas,
                        matrix=matrix,
                        model_dir=model_dir,
                        h5_path=sample_output_dir / grad_h5.name,
                        layout=layout,
                        adapter_fields=adapter_fields,
                    ),
                )
                row.update(
                    {
                        "status": "predicted",
                        "prediction_path": str(npz_path),
                        "metadata_path": str(json_path),
                        "nnz": int(matrix.nnz),
                        "shape_rows": int(matrix.shape[0]),
                        "shape_cols": int(matrix.shape[1]),
                        "error": "",
                    }
                )
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

    manifest = _manifest_payload(args, stencil_root, output_root, model_dir, requests, rows, started_at)
    manifest["runtime"] = {
        "mode": "deeph_autograd_backend",
        "preprocess_command": " ".join(preprocess_record["command"]),
        "preprocess_returncode": int(preprocess_record["returncode"]),
        "preprocess_stdout": preprocess_record["stdout"],
        "preprocess_stderr": preprocess_record["stderr"],
        "inference_cli": inference_cli,
        "inference_records": inference_records,
        "python_interpreter": str(settings["python_interpreter"]),
        "cwd": str(command_cwd),
        "pythonpath_prefix": str(infer_deeph_source_repo(args.deeph_command) or ""),
    }
    _write_outputs(output_root, rows, manifest)
    failed = [row for row in rows if row["status"] == "error"]
    if failed:
        raise DeepHAutogradDerivativePredictionError(
            f"DeepH autograd derivative prediction failed for {len(failed)} pair(s). "
            f"See {output_root / MANIFEST_FILENAME}"
        )
    return manifest


def _deeph_base_equivalence_fields(
    *,
    work_dir: Path,
    processed_sample_dir: Path,
    sample_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        adapter_result = adapt_deeph_prediction_sample(
            work_dir=work_dir,
            processed_sample_dir=processed_sample_dir,
            sample_id=sample_id,
            prediction_filename="hamiltonians_pred.h5",
        )
        write_json(output_dir / "deeph_adapter_result.json", adapter_result.to_dict())
        return {
            **adapter_result.metric_fields(),
            "claim_status": "diagnostic_only" if adapter_result.diagnostic_only else "raw_global_equivalence_proven",
        }
    except Exception as exc:
        return {
            "claim_status": "diagnostic_only",
            "deeph_diagnostic_only": True,
            "deeph_diagnostic_reason": f"deeph_base_equivalence_adapter_failed:{type(exc).__name__}",
            "deeph_equivalence_status": "unproven",
            "deeph_equivalence_scope": "unknown",
            "deeph_equivalence_reason": str(exc),
            "deeph_raw_global_equivalence_proven": False,
        }


def _manifest_payload(
    args: argparse.Namespace,
    stencil_root: Path,
    output_root: Path,
    model_dir: Path,
    requests: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    failed = [row for row in rows if row["status"] == "error"]
    return {
        "schema_version": "derivative_deeph_autograd_prediction_v1",
        "stencil_root": str(stencil_root),
        "output_root": str(output_root),
        "model": "deeph",
        "deeph_prediction_method": DEEPH_PREDICTION_METHOD_AUTOGRAD,
        "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
        "predicted_derivative_method": PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
        "predicted_delta_ang": None,
        "topology_fixed": True,
        "model_dir": str(model_dir),
        "hamiltonian_units": EXPECTED_HAMILTONIAN_UNITS,
        "displacement_units": EXPECTED_DISPLACEMENT_UNITS,
        "derivative_units": EXPECTED_DERIVATIVE_UNITS,
        "base_structures_total": len(requests),
        "samples_total": len(rows),
        "samples_ok": len([row for row in rows if row["status"] in {"predicted", "skipped_existing"}]),
        "samples_failed": len(failed),
        "elapsed_seconds": time.time() - started_at,
        "overwrite": bool(args.overwrite),
        "skip_if_exists": bool(args.skip_if_exists),
        "rows": rows,
        "outputs": {
            "status_csv": str(output_root / STATUS_FILENAME),
            "manifest": str(output_root / MANIFEST_FILENAME),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stencil-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--deeph-command", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-sample-id", action="append", default=[])
    parser.add_argument("--atoms", type=int, action="append", default=[])
    parser.add_argument("--axes", action="append", default=[])
    parser.add_argument("--max-base-structures", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        manifest = run_deeph_autograd_derivative_predictions(args)
    except (DeepHAutogradDerivativePredictionError, DerivativePredictionStageError) as exc:
        print(f"[DEEPH-AUTOGRAD-DERIVATIVE][ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "samples_total": manifest["samples_total"],
                "samples_ok": manifest["samples_ok"],
                "samples_failed": manifest["samples_failed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
