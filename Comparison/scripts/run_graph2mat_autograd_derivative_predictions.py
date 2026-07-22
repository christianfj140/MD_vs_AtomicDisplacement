#!/usr/bin/env python3
"""Direct Graph2Mat dH_pred/dR predictions via vectorized autograd.

For each base (undisplaced) stencil structure this script loads the Graph2Mat
model once, computes the jacobian of the predicted labels with respect to all
atomic positions with a vectorized strategy, and serializes one sparse
``dH_pred/dR_atom,axis`` matrix per stencil-required (atom, axis) pair.

Unlike ``run_hamiltonian_derivative_predictions.py`` (finite-difference route,
kept unchanged), no ML predictions are generated for the displaced R+delta /
R-delta structures: the derivative comes directly from the model loaded in
memory, evaluated at the base geometry with fixed neighbor topology.

Outputs, per base structure sample::

    <output_root>/<base_structure_sample_id>/dH_pred_atom{A}_axis{I}.npz
    <output_root>/<base_structure_sample_id>/dH_pred_atom{A}_axis{I}.json

where the ``.npz`` holds a scipy CSR matrix in the same
``(n_orbitals, n_orbitals * n_supercells)`` layout that ``ML_prediction.HSX``
files load into, and the ``.json`` holds unambiguous provenance metadata
(``predicted_delta_ang`` is null: the direct derivative has no stencil delta).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"
if str(TORCH_COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_COMPAT_DIR))
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402
from graph2mat_material_config import (  # noqa: E402
    resolve_matrix_component_policy,
    validate_model_matrix_component_policy,
)
from hamiltonian_derivative_stencil import (  # noqa: E402
    EXPECTED_DERIVATIVE_UNITS,
    EXPECTED_DISPLACEMENT_UNITS,
    EXPECTED_HAMILTONIAN_UNITS,
    GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD,
    PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
    REFERENCE_DERIVATIVE_METHOD_SIESTA,
    VALID_AXES,
    direct_derivative_prediction_basename,
    sparse_blockwise_hermiticity_defect,
    sparse_hermiticity_defect,
)
from predict_model_on_dataset import (  # noqa: E402
    checkpoint_training_dir,
    normalize_pattern_for_workdir,
)
from run_inventory import collect_run_inventory  # noqa: E402
from artifact_signature import (  # noqa: E402
    CACHE_VALID,
    cached_result_status,
    input_signature_sha256,
)
from artifact_signature import file_sha256 as sig_file_sha256  # noqa: E402

allow_graph2mat_checkpoint_globals()

MANIFEST_FILENAME = "derivative_graph2mat_autograd_prediction_manifest.json"
STATUS_FILENAME = "derivative_graph2mat_autograd_prediction_status.csv"
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


class AutogradDerivativePredictionError(RuntimeError):
    """Raised when the autograd derivative prediction stage fails closed."""


def expand_repo_tokens(text: str) -> str:
    return os.path.expandvars(text.replace("${REPO_ROOT}", str(REPO_ROOT)))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
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
        raise AutogradDerivativePredictionError(
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
            raise AutogradDerivativePredictionError(
                f"Unreadable stencil metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise AutogradDerivativePredictionError(
                f"Stencil metadata must be a JSON object: {metadata_path}"
            )
        payload = dict(payload)
        payload["_sample_dir"] = sample_dir
        payload["_structure_sample_id"] = sample_dir.name
        payloads.append(payload)
    if not payloads:
        raise AutogradDerivativePredictionError(
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
    """Group stencil metadata into per-base-structure derivative requests.

    Each request maps one base structure directory to the unique
    (atom_index_zero_based, axis) pairs required by its displaced samples and
    the reference stencil deltas they use.
    """

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
        pair_key = (int(atom_index), axis)
        deltas = request["pairs"].setdefault(pair_key, set())
        try:
            delta_value = float(delta_ang)
        except (TypeError, ValueError):
            delta_value = None
        if delta_value is not None and delta_value > 0:
            deltas.add(delta_value)

    if missing_bases:
        raise AutogradDerivativePredictionError(
            "Autograd derivative predictions require the base (undisplaced) stencil "
            "structures, but they are missing for: " + ", ".join(sorted(missing_bases))
            + ". Rebuild the stencils with --include-base."
        )
    if not requests:
        raise AutogradDerivativePredictionError(
            "No displaced stencil samples matched the requested base/atom/axis filters."
        )
    ordered = [requests[key] for key in sorted(requests)]
    if max_base_structures is not None:
        ordered = ordered[: max(0, int(max_base_structures))]
    if not ordered:
        raise AutogradDerivativePredictionError("No base structures selected.")
    return ordered


def _prediction_file_metadata(
    *,
    request: dict[str, Any],
    atom_index: int,
    axis: str,
    reference_deltas: list[float],
    matrix: sparse.csr_matrix,
    jacobian_method: str,
    jacobian_chunk_size: int | None,
    checkpoint: Path,
    n_atoms: int,
    n_outputs: int,
    translation_sum_rule: dict[str, float] | None = None,
    supercell_order: list | None = None,
) -> dict[str, Any]:
    axis_index = VALID_AXES[axis]
    hermiticity = sparse_hermiticity_defect(matrix)
    blockwise_hermiticity = (
        sparse_blockwise_hermiticity_defect(matrix, supercell_order)
        if supercell_order
        else math.nan
    )
    return {
        "schema": "graph2mat_autograd_direct_derivative_v1",
        "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
        "predicted_derivative_method": PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
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
        "graph2mat_prediction_method": GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD,
        "jacobian_method": jacobian_method,
        "jacobian_chunk_size": jacobian_chunk_size,
        "topology_fixed": True,
        "positions_frame_note": (
            "The jacobian differentiates w.r.t. batch-frame positions (e3nn "
            "change of basis); the cartesian derivative was obtained by chain "
            "rule with basis_table.change_of_basis, so axis_index refers to "
            "the physical SIESTA/fdf cartesian axis."
        ),
        "change_of_basis_applied": True,
        "base_sample_id": request["base_sample_id"],
        "base_structure_sample_id": request["base_structure_sample_id"],
        "split": request.get("split"),
        "checkpoint": str(checkpoint),
        "n_atoms": n_atoms,
        "n_outputs": n_outputs,
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "dh_hermiticity_defect": None if not math.isfinite(hermiticity) else hermiticity,
        "dh_blockwise_hermiticity_defect": (
            None if not math.isfinite(blockwise_hermiticity) else blockwise_hermiticity
        ),
        "translation_sum_rule": translation_sum_rule,
        "hermiticity_note": (
            "Measured, not enforced. dh_hermiticity_defect is NaN/null for rectangular "
            "(supercell) layouts; dh_blockwise_hermiticity_defect pairs D(R) with "
            "D(-R)^dagger over the supercell columns instead."
        ),
    }


def _append_missing_structure_rows(
    rows: list[dict[str, Any]],
    requests_by_structure_id: dict[str, dict[str, Any]],
    seen_structure_ids: set[str],
) -> None:
    for structure_id, request in sorted(requests_by_structure_id.items()):
        if structure_id in seen_structure_ids:
            continue
        for (atom_index, axis), deltas in sorted(request["pairs"].items()):
            axis_index = VALID_AXES[axis]
            rows.append(
                {
                    "base_structure_sample_id": structure_id,
                    "base_sample_id": request["base_sample_id"],
                    "atom_index_zero_based": atom_index,
                    "axis": axis,
                    "axis_index": axis_index,
                    "status": "error",
                    "prediction_path": "",
                    "metadata_path": "",
                    "nnz": None,
                    "shape_rows": None,
                    "shape_cols": None,
                    "reference_delta_ang_values": ";".join(f"{d:g}" for d in sorted(deltas)),
                    "error": "missing_base_structure_from_graph2mat_dataloader",
                }
            )


def run_autograd_derivative_predictions(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    stencil_root = args.stencil_root.expanduser().resolve(strict=False)
    output_root = (
        args.output_root.expanduser().resolve(strict=False)
        if args.output_root is not None
        else stencil_root / "predicted_derivative_hamiltonians"
    )
    checkpoint = args.checkpoint.expanduser().resolve(strict=False)
    if not checkpoint.exists():
        raise AutogradDerivativePredictionError(
            f"Graph2Mat autograd derivative prediction requires an existing --checkpoint: {checkpoint}"
        )
    if not args.basis_files:
        raise AutogradDerivativePredictionError(
            "Graph2Mat autograd derivative prediction requires --basis-files "
            "(same basis XML glob used by the finite-difference route)."
        )
    args.basis_files = expand_repo_tokens(str(args.basis_files))

    atoms = [int(atom) for atom in (args.atoms or [])]
    axes = [str(axis).strip().lower() for axis in (args.axes or [])]
    invalid_axes = [axis for axis in axes if axis not in VALID_AXES]
    if invalid_axes:
        raise AutogradDerivativePredictionError(
            f"Unsupported axes: {', '.join(invalid_axes)}. Use x, y, z."
        )

    payloads = read_structure_metadata(stencil_root / "structures")
    requests = collect_derivative_requests(
        payloads,
        base_sample_ids=[str(item) for item in (args.base_sample_id or [])],
        atoms=atoms,
        axes=axes,
        max_base_structures=args.max_base_structures,
    )

    run_inventory = collect_run_inventory()
    base_signature_payload = {
        "model": "graph2mat",
        "checkpoint_sha256": sig_file_sha256(checkpoint),
        "repository_commits": {
            name: state.get("commit") for name, state in run_inventory["repositories"].items()
        },
        "repository_dirty_states": {
            name: state.get("dirty") for name, state in run_inventory["repositories"].items()
        },
        "basis_files": str(args.basis_files),
        "matrix_component_policy": str(args.matrix_component_policy or ""),
        "derivative_method": PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
        "jacobian_method": str(args.jacobian_method),
        "dtype": "float64",
        "topology_fixed": True,
    }

    def _pair_signature(request: dict[str, Any], atom_index: int, axis_index: int) -> str:
        return input_signature_sha256(
            {
                **base_signature_payload,
                "structure_fdf_sha256": sig_file_sha256(
                    Path(request["base_sample_dir"]) / "RUN.fdf"
                ),
                "atom_index": int(atom_index),
                "axis_index": int(axis_index),
            }
        )

    matrix_component_policy, n_matrix_components = resolve_matrix_component_policy(
        {
            "matrix_component_policy": args.matrix_component_policy,
            "n_matrix_components": args.n_matrix_components,
        },
        context="autograd derivative prediction CLI",
    )

    import inspect

    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from graph2mat import AtomicTableWithEdges

    from graph2mat_autograd_derivatives import (
        Graph2MatAutogradDerivativeError,
        compute_graph2mat_position_jacobian,
        derivative_prediction_to_sparse_matrices,
        select_derivative_prediction_from_jacobian,
        translation_sum_rule_metrics,
    )

    allow_graph2mat_checkpoint_globals()
    if args.accelerator == "cuda":
        # Verified failure on torch 2.11 + mace 0.3.15: the vectorized batched
        # backward (torch.autograd.grad with is_grads_batched=True) aborts
        # inside MACE's TorchScript-compiled modules on CUDA ("Cannot access
        # data pointer of Tensor that doesn't have storage"), and disabling
        # e3nn's jit_script_fx is not sufficient because MACE scripts its own
        # submodules. Fail closed instead of silently producing nothing.
        raise AutogradDerivativePredictionError(
            "--accelerator cuda is not supported: MACE's TorchScript modules "
            "reject the vectorized batched backward on CUDA with current "
            "torch/mace versions. Run on cpu (default)."
        )
    device = torch.device("cpu")

    # The checkpoint's saved basis_files hparam is a path relative to the cwd
    # used at training time, which does not always match the directory
    # layout reconstructed by checkpoint_training_dir() (e.g. after a run
    # root was moved or restructured). Build the basis table directly from
    # the real --basis-files CLI arg (absolute, always correct) and pass it
    # as a load_from_checkpoint override instead of relying on that glob.
    basis_table = AtomicTableWithEdges.from_basis_glob(
        Path(args.basis_files).parent.glob(Path(args.basis_files).name)
    )
    run_cwd = checkpoint_training_dir(checkpoint)
    source_cwd = Path.cwd()
    basis_files = normalize_pattern_for_workdir(
        str(args.basis_files), source_cwd=source_cwd, target_cwd=run_cwd
    )

    runs_payload = {
        "predict": [str(request["base_sample_dir"] / "RUN.fdf") for request in requests]
    }
    runs_json_path = output_root / ".autograd_predict_runs.json"
    write_json(runs_json_path, runs_payload)

    datamodule_kwargs: dict[str, Any] = {
        "out_matrix": args.out_matrix,
        "symmetric_matrix": bool(args.symmetric_matrix),
        "sub_point_matrix": bool(args.sub_point_matrix),
        "basis_files": basis_files,
        "runs_json": str(runs_json_path),
        "store_in_memory": True,
        "batch_size": 1,
    }
    signature = inspect.signature(MatrixDataModule)
    if "n_matrix_components" not in signature.parameters or "matrix_component_policy" not in signature.parameters:
        raise AutogradDerivativePredictionError(
            "MatrixDataModule does not expose n_matrix_components/matrix_component_policy; "
            "this benchmark cannot enforce H-only target semantics with this Graph2Mat version."
        )
    datamodule_kwargs["n_matrix_components"] = n_matrix_components
    datamodule_kwargs["matrix_component_policy"] = matrix_component_policy
    if "loader_threads" in signature.parameters and args.loader_threads is not None:
        datamodule_kwargs["loader_threads"] = int(args.loader_threads)

    requests_by_structure_id = {
        request["base_structure_sample_id"]: request for request in requests
    }

    rows: list[dict[str, Any]] = []
    seen_structure_ids: set[str] = set()
    started_at = time.time()
    cwd = Path.cwd()
    try:
        os.chdir(run_cwd)
        model = LitMACEMatrixModel.load_from_checkpoint(
            str(checkpoint), weights_only=False, basis_table=basis_table
        )
        validate_model_matrix_component_policy(
            model,
            matrix_component_policy=matrix_component_policy,
            n_matrix_components=n_matrix_components,
            context="autograd derivative prediction",
        )
        model = model.to(device)
        model.eval()
        datamodule = MatrixDataModule(**datamodule_kwargs)
        datamodule.setup("predict")
        data_processor = datamodule.data_processor
        loader = datamodule.predict_dataloader()

        for batch in loader:
            paths = batch.metadata.get("path") if isinstance(batch.metadata, dict) else None
            if not paths:
                raise AutogradDerivativePredictionError(
                    "Prediction batch does not expose its source structure path; "
                    "cannot map the jacobian to a base stencil sample."
                )
            structure_id = Path(str(paths[0])).parent.name
            request = requests_by_structure_id.get(structure_id)
            if request is None:
                raise AutogradDerivativePredictionError(
                    f"Prediction batch structure {structure_id!r} does not match any "
                    "requested base stencil sample."
                )
            seen_structure_ids.add(structure_id)
            batch = batch.to(device) if hasattr(batch, "to") else batch

            jacobian_result = compute_graph2mat_position_jacobian(
                model.model,
                batch,
                method=args.jacobian_method,
                chunk_size=args.jacobian_chunk_size,
            )
            translation_sum_rule = translation_sum_rule_metrics(jacobian_result.jacobian)
            # Sparse serialization goes through numpy/sisl; keep a CPU batch.
            serialization_batch = batch.to("cpu") if device.type != "cpu" else batch

            for (atom_index, axis), deltas in sorted(request["pairs"].items()):
                axis_index = VALID_AXES[axis]
                reference_deltas = sorted(deltas)
                basename = direct_derivative_prediction_basename(atom_index, axis_index)
                npz_path = output_root / structure_id / f"{basename}.npz"
                json_path = output_root / structure_id / f"{basename}.json"
                row = {
                    "base_structure_sample_id": structure_id,
                    "base_sample_id": request["base_sample_id"],
                    "atom_index_zero_based": atom_index,
                    "axis": axis,
                    "axis_index": axis_index,
                    "status": "error",
                    "prediction_path": "",
                    "metadata_path": "",
                    "nnz": None,
                    "shape_rows": None,
                    "shape_cols": None,
                    "reference_delta_ang_values": ";".join(f"{d:g}" for d in reference_deltas),
                    "error": "",
                }
                try:
                    pair_signature = _pair_signature(request, atom_index, axis_index)
                    if npz_path.exists() and args.skip_if_exists and not args.overwrite:
                        cache_status = cached_result_status(npz_path, json_path, pair_signature)
                        if cache_status == CACHE_VALID:
                            matrix = sparse.load_npz(npz_path)
                            row.update(
                                {
                                    "status": "skipped_existing",
                                    "cache_status": cache_status,
                                    "prediction_path": str(npz_path),
                                    "metadata_path": str(json_path),
                                    "nnz": int(matrix.nnz),
                                    "shape_rows": int(matrix.shape[0]),
                                    "shape_cols": int(matrix.shape[1]),
                                }
                            )
                            rows.append(row)
                            continue
                        # Stale/legacy/mismatched cache entries are recomputed.
                        row["cache_status"] = cache_status
                    derivative_prediction = select_derivative_prediction_from_jacobian(
                        jacobian_result.jacobian,
                        jacobian_result.spec,
                        atom_index,
                        axis_index,
                        change_of_basis=data_processor.basis_table.change_of_basis,
                    )
                    supercell_orders: list = []
                    matrices = derivative_prediction_to_sparse_matrices(
                        data_processor,
                        serialization_batch,
                        derivative_prediction,
                        supercell_orders=supercell_orders,
                    )
                    if len(matrices) != 1:
                        raise AutogradDerivativePredictionError(
                            f"Expected one sparse matrix per base structure, got {len(matrices)}."
                        )
                    matrix = matrices[0].astype(np.float64)
                    npz_path.parent.mkdir(parents=True, exist_ok=True)
                    with npz_path.open("wb") as handle:
                        sparse.save_npz(handle, matrix)
                    metadata_payload = _prediction_file_metadata(
                        request=request,
                        atom_index=atom_index,
                        axis=axis,
                        reference_deltas=reference_deltas,
                        matrix=matrix,
                        jacobian_method=jacobian_result.method,
                        jacobian_chunk_size=jacobian_result.chunk_size,
                        checkpoint=checkpoint,
                        n_atoms=jacobian_result.n_atoms,
                        n_outputs=jacobian_result.spec.n_outputs,
                        translation_sum_rule=translation_sum_rule,
                        supercell_order=supercell_orders[0] if supercell_orders else None,
                    )
                    metadata_payload["input_signature_sha256"] = pair_signature
                    write_json(json_path, metadata_payload)
                    row.update(
                        {
                            "status": "predicted",
                            "prediction_path": str(npz_path),
                            "metadata_path": str(json_path),
                            "nnz": int(matrix.nnz),
                            "shape_rows": int(matrix.shape[0]),
                            "shape_cols": int(matrix.shape[1]),
                        }
                    )
                except (Graph2MatAutogradDerivativeError, AutogradDerivativePredictionError, ValueError) as exc:
                    row["error"] = str(exc)
                rows.append(row)
    finally:
        os.chdir(cwd)

    _append_missing_structure_rows(rows, requests_by_structure_id, seen_structure_ids)
    failed = [row for row in rows if row["status"] == "error"]
    manifest = {
        "schema_version": "derivative_graph2mat_autograd_prediction_v1",
        "stencil_root": str(stencil_root),
        "output_root": str(output_root),
        "model": "graph2mat",
        "graph2mat_prediction_method": GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD,
        "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
        "predicted_derivative_method": PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
        "predicted_delta_ang": None,
        "topology_fixed": True,
        "checkpoint": str(checkpoint),
        "basis_files": str(args.basis_files),
        "accelerator": args.accelerator,
        "jacobian_method": args.jacobian_method,
        "jacobian_chunk_size": args.jacobian_chunk_size,
        "matrix_component_policy": matrix_component_policy,
        "n_matrix_components": n_matrix_components,
        "hamiltonian_units": EXPECTED_HAMILTONIAN_UNITS,
        "displacement_units": EXPECTED_DISPLACEMENT_UNITS,
        "derivative_units": EXPECTED_DERIVATIVE_UNITS,
        "base_structures_total": len(requests),
        "samples_total": len(rows),
        "samples_ok": len([row for row in rows if row["status"] in {"predicted", "skipped_existing"}]),
        "samples_failed": len(failed),
        "elapsed_seconds": time.time() - started_at,
        "run_inventory": run_inventory,
        "rows": rows,
        "outputs": {
            "status_csv": str(output_root / STATUS_FILENAME),
            "manifest": str(output_root / MANIFEST_FILENAME),
        },
    }
    write_csv(output_root / STATUS_FILENAME, rows)
    write_json(output_root / MANIFEST_FILENAME, _json_safe(manifest))
    if failed:
        raise AutogradDerivativePredictionError(
            f"Autograd derivative prediction failed for {len(failed)} pair(s). "
            f"See {output_root / MANIFEST_FILENAME}"
        )
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stencil-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--basis-files", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--accelerator", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out-matrix", default="hamiltonian")
    parser.add_argument("--symmetric-matrix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sub-point-matrix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--matrix-component-policy", default="h_only")
    parser.add_argument("--n-matrix-components", type=int, default=1)
    parser.add_argument("--loader-threads", type=int, default=None)
    parser.add_argument(
        "--jacobian-method",
        choices=["auto", "vmap_vjp_chunked", "jacrev", "jacfwd", "autograd_jacobian"],
        default="auto",
        help=(
            "Vectorized jacobian strategy. 'auto' uses chunked batched VJPs, the "
            "only strategy compatible with current MACE models (functorch "
            "transforms are rejected by MACE's in-forward requires_grad_)."
        ),
    )
    parser.add_argument(
        "--jacobian-chunk-size",
        type=int,
        default=None,
        help="Cotangent chunk size for chunked strategies (default 64; memory scales with it).",
    )
    parser.add_argument("--base-sample-id", action="append", default=[])
    parser.add_argument("--atoms", type=int, action="append", default=[])
    parser.add_argument("--axes", action="append", default=[])
    parser.add_argument("--max-base-structures", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-if-exists", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        manifest = run_autograd_derivative_predictions(args)
    except AutogradDerivativePredictionError as exc:
        print(f"[AUTOGRAD-DERIVATIVE-PREDICT][ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "samples_total": manifest["samples_total"],
                "samples_ok": manifest["samples_ok"],
                "samples_failed": manifest["samples_failed"],
                "base_structures_total": manifest["base_structures_total"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
