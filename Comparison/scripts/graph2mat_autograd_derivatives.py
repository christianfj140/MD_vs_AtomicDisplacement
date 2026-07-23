#!/usr/bin/env python3
"""Vectorized autograd derivatives of Graph2Mat outputs w.r.t. atomic positions.

This module computes ``J = d outputs_flat / d positions`` for a Graph2Mat
model loaded in memory, where ``outputs_flat`` is the concatenation of the
flattened ``node_labels`` and ``edge_labels`` predictions and ``positions``
are the cartesian atomic positions of a single structure.

Units
-----
Graph2Mat predicts Hamiltonian labels in eV from positions in Ang, so the
jacobian (and any derivative selected from it) is directly in eV/Ang. No unit
conversion is applied anywhere in this module.

Fixed neighbor topology
-----------------------
``edge_index`` and ``shifts`` are taken from the input batch and kept constant
inside the differentiable closure. The jacobian is therefore the local
derivative of the model for the fixed neighbor topology of the base structure
(exact up to cutoff-induced connectivity changes, which are discontinuous
anyway).

Method trade-offs (measured on graphene 5x5, 50 atoms, 12800 outputs)
---------------------------------------------------------------------
``vmap_vjp_chunked`` (default)
    Classic autograd VJPs batched with ``torch.autograd.grad(...,
    is_grads_batched=True)`` over chunks of one-hot cotangents. Works with the
    real MACE-backed model and bounds memory: backward activations scale with
    ``chunk_size`` (chunk 512 exhausted 44 GiB on graphene 5x5; chunk 64 is
    safe). The final dense jacobian itself is tiny (~0.007 GiB float32).
``jacrev`` / ``jacfwd``
    torch.func transforms. *Currently incompatible with MACE >= 0.3.x*: MACE's
    ``prepare_graph`` calls ``data["positions"].requires_grad_(True)`` inside
    the forward, which functorch transforms reject (RuntimeError). They remain
    available for functorch-compatible models (e.g. the fake models used in
    unit tests). ``jacfwd`` would otherwise be the natural choice here because
    ``n_inputs = n_atoms * 3`` (150) is much smaller than ``n_outputs``.
``autograd_jacobian``
    ``torch.autograd.functional.jacobian(vectorize=True)``. Compatible with
    MACE, but it batches *all* ``n_outputs`` cotangents in a single backward,
    so memory explodes for realistic output counts. Only for small problems.
"""

from __future__ import annotations

import sys
from copy import copy as _shallow_copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_OUTPUT_KEYS = ("node_labels", "edge_labels")
JACOBIAN_METHODS = (
    "auto",
    "jvp_double_backward",
    "vmap_vjp_chunked",
    "jacrev",
    "jacfwd",
    "autograd_jacobian",
)
# Backward-activation memory scales linearly with the cotangent chunk;
# 64 was measured safe on CPU for graphene 5x5 (512 triggered the OOM killer).
DEFAULT_JACOBIAN_CHUNK_SIZE = 64


class Graph2MatAutogradDerivativeError(RuntimeError):
    """Raised when the autograd derivative route cannot proceed safely."""


@dataclass(frozen=True)
class PredictionFlattenSpec:
    """Slicing/shape metadata to rebuild label dicts from a flat vector."""

    output_keys: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    slices: dict[str, tuple[int, int]]
    n_outputs: int


@dataclass(frozen=True)
class PositionJacobianResult:
    jacobian: torch.Tensor
    spec: PredictionFlattenSpec
    base_predictions: dict[str, torch.Tensor]
    method: str
    chunk_size: int | None
    n_atoms: int


def flatten_graph2mat_predictions(
    predictions: Dict[str, torch.Tensor],
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
) -> tuple[torch.Tensor, PredictionFlattenSpec]:
    """Concatenate prediction tensors into a flat vector plus rebuild metadata."""

    flats: list[torch.Tensor] = []
    shapes: dict[str, tuple[int, ...]] = {}
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in output_keys:
        if key not in predictions:
            raise Graph2MatAutogradDerivativeError(
                f"Prediction output {key!r} is missing; available keys: {sorted(predictions)}"
            )
        tensor = predictions[key]
        if not isinstance(tensor, torch.Tensor):
            raise Graph2MatAutogradDerivativeError(
                f"Prediction output {key!r} must be a torch.Tensor, got {type(tensor).__name__}."
            )
        flat = tensor.reshape(-1)
        shapes[key] = tuple(tensor.shape)
        slices[key] = (offset, offset + flat.numel())
        offset += flat.numel()
        flats.append(flat)
    outputs_flat = torch.cat(flats) if flats else torch.empty(0)
    spec = PredictionFlattenSpec(
        output_keys=tuple(output_keys),
        shapes=shapes,
        slices=slices,
        n_outputs=offset,
    )
    return outputs_flat, spec


def unflatten_graph2mat_prediction_vector(
    flat_vector: torch.Tensor,
    spec: PredictionFlattenSpec,
) -> dict[str, torch.Tensor]:
    """Rebuild a predictions dict with the original keys/shapes from a flat vector."""

    if flat_vector.numel() != spec.n_outputs:
        raise Graph2MatAutogradDerivativeError(
            f"Flat vector has {flat_vector.numel()} elements, expected {spec.n_outputs}."
        )
    return {
        key: flat_vector[spec.slices[key][0] : spec.slices[key][1]].reshape(spec.shapes[key])
        for key in spec.output_keys
    }


def _clone_batch(batch: Any) -> Any:
    if hasattr(batch, "clone"):
        return batch.clone()
    if isinstance(batch, dict):
        return dict(batch)
    return _shallow_copy(batch)


def _batch_num_structures(batch: Any) -> int | None:
    num_graphs = getattr(batch, "num_graphs", None)
    if num_graphs is not None:
        return int(num_graphs)
    ptr = None
    try:
        ptr = batch["ptr"]
    except (KeyError, TypeError, IndexError):
        ptr = getattr(batch, "ptr", None)
    if ptr is not None:
        return max(int(len(ptr)) - 1, 0)
    return None


def require_single_structure_batch(batch: Any) -> None:
    """The derivative route maps one jacobian to one structure; reject batches > 1."""

    num_structures = _batch_num_structures(batch)
    if num_structures is not None and num_structures != 1:
        raise Graph2MatAutogradDerivativeError(
            "Autograd derivative predictions require batch_size=1 (one structure "
            f"per jacobian), got a batch with {num_structures} structures."
        )


def graph2mat_forward_labels(
    model: torch.nn.Module,
    batch: Any,
    positions: torch.Tensor,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
) -> tuple[torch.Tensor, PredictionFlattenSpec]:
    """Differentiable forward pass with ``positions`` substituted into the batch.

    Never uses ``torch.no_grad()``/``torch_predict()`` and never detaches or
    converts to numpy: the returned flat tensor stays connected to
    ``positions`` in the autograd graph. ``edge_index`` and ``shifts`` are
    left untouched (fixed topology).
    """

    model.eval()
    with torch.set_grad_enabled(True):
        data = _clone_batch(batch)
        data["positions"] = positions
        out = model(data)
        return flatten_graph2mat_predictions(out, output_keys=output_keys)


def _jacobian_vmap_vjp_chunked(
    forward: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full jacobian via chunked batched VJPs (classic autograd + vmap backward).

    Chunks run over blocks of ``outputs_flat`` cotangents, never over single
    elements: each ``torch.autograd.grad(..., is_grads_batched=True)`` call is
    one vectorized backward for ``chunk_size`` one-hot cotangents.
    """

    positions_leaf = positions.detach().clone().requires_grad_(True)
    outputs = forward(positions_leaf)
    if not outputs.requires_grad:
        raise Graph2MatAutogradDerivativeError(
            "Model outputs are detached from positions; check for torch.no_grad()/"
            "detach() inside the forward pass."
        )
    n_outputs = outputs.numel()
    rows: list[torch.Tensor] = []
    for start in range(0, n_outputs, chunk_size):
        stop = min(start + chunk_size, n_outputs)
        basis = torch.zeros(
            (stop - start, n_outputs), dtype=outputs.dtype, device=outputs.device
        )
        basis[torch.arange(stop - start), torch.arange(start, stop)] = 1.0
        (vjp,) = torch.autograd.grad(
            outputs,
            positions_leaf,
            grad_outputs=basis,
            retain_graph=stop < n_outputs,
            is_grads_batched=True,
        )
        rows.append(vjp)
    jacobian = torch.cat(rows) if rows else torch.zeros((0, *positions.shape))
    return jacobian, outputs.detach()


def _jacobian_jvp_double_backward(
    forward: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    target_atoms: Sequence[int],
) -> torch.Tensor:
    """Forward-mode jacobian columns for a few atoms, via forward-over-reverse.

    The finite-displacement stencils only request ``dH/dR`` for a handful of
    (atom, axis) pairs (typically atom 0, axes x/y/z), so computing the FULL
    reverse-mode jacobian over all ``n_outputs`` cotangents is wasteful: it
    scales with the number of Hamiltonian elements (~1e4), one backward per
    chunk. Forward-mode scales with the number of requested input directions
    instead (``len(target_atoms) * 3``).

    MACE calls ``data["positions"].requires_grad_(True)`` internally, which is
    incompatible with ``torch.func`` transforms (jvp/jacfwd). So we build the
    JVP by hand as a double backward: ``g = u^T J`` (u a differentiable dummy
    cotangent) is differentiated w.r.t. ``u`` along a one-hot input tangent,
    which yields ``J @ tangent`` — one column of the jacobian — using only
    ``torch.autograd.grad``. Numerically identical to the VJP route (verified:
    rel err ~4e-7, float32 noise) at ~4000x less cost.

    Returns a ``[n_outputs, n_atoms, 3]`` jacobian where only the columns of
    ``target_atoms`` are filled; all other atom columns are left as zeros (they
    are never selected downstream). The translation sum rule, which needs all
    atoms, is therefore only meaningful when every atom is a target.
    """

    positions_leaf = positions.detach().clone().requires_grad_(True)
    outputs = forward(positions_leaf)
    if not outputs.requires_grad:
        raise Graph2MatAutogradDerivativeError(
            "Model outputs are detached from positions; check for torch.no_grad()/"
            "detach() inside the forward pass."
        )
    n_outputs = int(outputs.numel())
    n_atoms = int(positions.shape[0])
    jacobian = torch.zeros(
        (n_outputs, n_atoms, 3), dtype=outputs.dtype, device=outputs.device
    )
    # u^T J: a differentiable dummy cotangent whose gradient reconstructs J columns.
    dummy = torch.zeros(n_outputs, dtype=outputs.dtype, device=outputs.device, requires_grad=True)
    (vjp_of_dummy,) = torch.autograd.grad(outputs, positions_leaf, grad_outputs=dummy, create_graph=True)
    unique_atoms = sorted({int(a) for a in target_atoms})
    for atom in unique_atoms:
        if not (0 <= atom < n_atoms):
            raise Graph2MatAutogradDerivativeError(
                f"target atom {atom} is outside the structure with {n_atoms} atoms."
            )
        for axis in range(3):
            tangent = torch.zeros_like(positions_leaf)
            tangent[atom, axis] = 1.0
            # d(u^T J)/du contracted with the input tangent = J @ tangent = column.
            (column,) = torch.autograd.grad(
                vjp_of_dummy, dummy, grad_outputs=tangent, retain_graph=True
            )
            jacobian[:, atom, axis] = column
    return jacobian


def compute_graph2mat_position_jacobian(
    model: torch.nn.Module,
    batch: Any,
    *,
    method: str = "auto",
    chunk_size: int | None = None,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
    target_atoms: Sequence[int] | None = None,
) -> PositionJacobianResult:
    """Compute ``J = d outputs_flat / d positions`` for a single-structure batch.

    Returns a jacobian of shape ``[n_outputs, n_atoms, 3]`` (same dtype/device
    as the batch positions), the flatten spec needed to rebuild label dicts,
    and the non-derived base predictions.

    ``target_atoms`` lists the atom indices whose derivative columns are
    actually consumed downstream (the stencil's requested atoms). When set and
    ``method`` resolves to forward-mode (``auto``/``jvp_double_backward``), only
    those columns are computed — orders of magnitude cheaper than the full
    reverse-mode jacobian. ``None`` means "all atoms" (full jacobian).
    """

    if method not in JACOBIAN_METHODS:
        raise Graph2MatAutogradDerivativeError(
            f"Unsupported jacobian method {method!r}. Use one of: {', '.join(JACOBIAN_METHODS)}."
        )
    require_single_structure_batch(batch)

    positions = batch["positions"]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise Graph2MatAutogradDerivativeError(
            f"positions must have shape [n_atoms, 3], got {tuple(positions.shape)}."
        )
    n_atoms = int(positions.shape[0])

    # Non-derived forward: base predictions + flatten spec (shapes/slices).
    model.eval()
    with torch.no_grad():
        base_out = model(_clone_batch(batch))
    base_flat, spec = flatten_graph2mat_predictions(base_out, output_keys=output_keys)
    base_predictions = unflatten_graph2mat_prediction_vector(base_flat, spec)

    def forward(positions_tensor: torch.Tensor) -> torch.Tensor:
        flat, forward_spec = graph2mat_forward_labels(
            model, batch, positions_tensor, output_keys=output_keys
        )
        if forward_spec.n_outputs != spec.n_outputs:
            raise Graph2MatAutogradDerivativeError(
                "Differentiable forward produced a different number of outputs "
                f"({forward_spec.n_outputs}) than the base forward ({spec.n_outputs})."
            )
        return flat

    # auto now resolves to the forward-mode JVP route: it computes only the
    # requested atom columns instead of the full reverse-mode jacobian over all
    # ~1e4 Hamiltonian outputs (~4000x faster, verified numerically identical).
    resolved_method = "jvp_double_backward" if method == "auto" else method
    resolved_chunk: int | None = chunk_size
    all_atoms = tuple(range(n_atoms))
    resolved_target_atoms = tuple(all_atoms if target_atoms is None else target_atoms)

    if resolved_method == "jvp_double_backward":
        jacobian = _jacobian_jvp_double_backward(forward, positions, resolved_target_atoms)
        resolved_chunk = None
    elif resolved_method == "vmap_vjp_chunked":
        if resolved_chunk is None:
            resolved_chunk = DEFAULT_JACOBIAN_CHUNK_SIZE
        jacobian, _ = _jacobian_vmap_vjp_chunked(forward, positions, int(resolved_chunk))
    elif resolved_method == "jacrev":
        jacobian = torch.func.jacrev(forward, chunk_size=resolved_chunk)(
            positions.detach().clone()
        )
    elif resolved_method == "jacfwd":
        jacobian = torch.func.jacfwd(forward)(positions.detach().clone())
    elif resolved_method == "autograd_jacobian":
        jacobian = torch.autograd.functional.jacobian(
            forward, positions.detach().clone(), vectorize=True
        )
    else:  # pragma: no cover - guarded above
        raise Graph2MatAutogradDerivativeError(f"Unhandled jacobian method {resolved_method!r}.")

    expected_shape = (spec.n_outputs, n_atoms, 3)
    if tuple(jacobian.shape) != expected_shape:
        raise Graph2MatAutogradDerivativeError(
            f"Jacobian has shape {tuple(jacobian.shape)}, expected {expected_shape}."
        )

    return PositionJacobianResult(
        jacobian=jacobian,
        spec=spec,
        base_predictions=base_predictions,
        method=resolved_method,
        chunk_size=int(resolved_chunk) if resolved_chunk is not None else None,
        n_atoms=n_atoms,
    )


def select_derivative_prediction_from_jacobian(
    jacobian: torch.Tensor,
    spec: PredictionFlattenSpec,
    atom_index: int,
    axis_index: int,
    *,
    change_of_basis: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Rebuild ``d labels / d R_atom,axis`` as a predictions-shaped dict.

    ``axis_index`` refers to the *physical cartesian* axis (the frame of the
    SIESTA/fdf structure and of the finite-displacement stencils).

    Graph2Mat batches do NOT store positions in that frame: ``_sanitize_data``
    applies ``cartesian_to_basis`` (the e3nn spherical-harmonics change of
    basis, ``p_batch = C @ p_cart``), so the jacobian columns are derivatives
    w.r.t. *batch-frame* coordinates. By the chain rule, the derivative along
    cartesian axis ``a`` is the contraction ``J[:, atom, :] @ C[:, a]``.

    Pass ``change_of_basis = data_processor.basis_table.change_of_basis`` for
    physically meaningful cartesian derivatives; ``None`` selects the raw
    batch-frame column (only correct when the change of basis is the
    identity, e.g. fake models in unit tests).
    """

    if jacobian.ndim != 3 or jacobian.shape[2] != 3:
        raise Graph2MatAutogradDerivativeError(
            f"Jacobian must have shape [n_outputs, n_atoms, 3], got {tuple(jacobian.shape)}."
        )
    n_atoms = int(jacobian.shape[1])
    if not (0 <= int(atom_index) < n_atoms):
        raise Graph2MatAutogradDerivativeError(
            f"atom_index {atom_index} is outside the structure with {n_atoms} atoms."
        )
    if int(axis_index) not in (0, 1, 2):
        raise Graph2MatAutogradDerivativeError(f"axis_index must be 0, 1 or 2, got {axis_index}.")
    if change_of_basis is None:
        column = jacobian[:, int(atom_index), int(axis_index)]
    else:
        cob = torch.as_tensor(change_of_basis, dtype=jacobian.dtype, device=jacobian.device)
        if tuple(cob.shape) != (3, 3):
            raise Graph2MatAutogradDerivativeError(
                f"change_of_basis must be a 3x3 matrix, got {tuple(cob.shape)}."
            )
        direction = cob[:, int(axis_index)]
        column = jacobian[:, int(atom_index), :] @ direction
    return unflatten_graph2mat_prediction_vector(column, spec)


def translation_sum_rule_metrics(jacobian: torch.Tensor) -> dict[str, float]:
    """Global-translation invariance: sum_I dH/dR_{I,alpha} should vanish.

    Frame-independent (summing over atoms commutes with the e3nn change of
    basis), so it can be evaluated directly on the batch-frame jacobian.
    """
    if jacobian.ndim != 3 or jacobian.shape[2] != 3:
        raise Graph2MatAutogradDerivativeError(
            f"Jacobian must have shape [n_outputs, n_atoms, 3], got {tuple(jacobian.shape)}."
        )
    translation = jacobian.sum(dim=1)  # [n_outputs, 3]
    per_atom_norm = torch.linalg.norm(jacobian.reshape(jacobian.shape[0], -1))
    residual_norm = float(torch.linalg.norm(translation))
    return {
        "translation_residual_max_abs": float(translation.abs().max()),
        "translation_residual_frobenius": residual_norm,
        "translation_residual_relative": (
            residual_norm / float(per_atom_norm) if float(per_atom_norm) > 0 else float("nan")
        ),
    }


def supercell_order_from_sisl_matrix(sisl_matrix: Any) -> list[tuple[int, int, int]] | None:
    """R-vector ordering of a sisl matrix's supercell columns (or None)."""
    geometry = getattr(sisl_matrix, "geometry", None)
    lattice = getattr(geometry, "lattice", None) or getattr(geometry, "sc", None)
    sc_off = getattr(lattice, "sc_off", None)
    if sc_off is None:
        return None
    return [tuple(int(x) for x in vector) for vector in sc_off]


def derivative_prediction_to_sparse_matrices(
    data_processor: Any,
    batch: Any,
    derivative_prediction: dict[str, torch.Tensor],
    *,
    threshold: float | None = None,
    supercell_orders: list[list[tuple[int, int, int]] | None] | None = None,
) -> list[Any]:
    """Convert derivative labels to sparse matrices via the existing mapping.

    ``supercell_orders``, when passed as an empty list, is filled with one
    R-vector ordering (sisl ``sc_off``) per returned matrix, for real-space
    blockwise hermiticity checks on the rectangular supercell layout.

    Reuses ``data_processor.yield_from_batch`` so the orbital/block mapping and
    the symmetric-edge accounting are exactly the ones used for normal
    Graph2Mat predictions. Two derivative-specific adjustments:

    - ``sub_point_matrix`` is forced off for the conversion: ``labels_to``
      adds the constant per-species point matrix back to node labels, and a
      constant has zero derivative, so adding it would corrupt dH/dR.
    - ``threshold`` defaults to ``None`` (keep every entry) instead of the
      1e-8 used for absolute Hamiltonians, because derivative magnitudes are
      not on the same scale as H.

    Returns one ``scipy.sparse.csr_matrix`` per structure in the batch, in the
    same (no, no * n_supercells) layout that ``ML_prediction.HSX`` files load
    into via ``sisl`` + ``tocsr(0)``.
    """

    predictions = {
        key: value.detach().to("cpu") for key, value in derivative_prediction.items()
    }
    conversion_processor = data_processor
    if getattr(data_processor, "sub_point_matrix", False):
        conversion_processor = data_processor.copy(sub_point_matrix=False)

    matrices = []
    for example in conversion_processor.yield_from_batch(
        batch, predictions=predictions, as_matrix=False
    ):
        sisl_matrix = conversion_processor.labels_to(
            conversion_processor.default_out_format,
            data=example,
            threshold=threshold,
        )
        csr = sisl_matrix.tocsr(0) if hasattr(sisl_matrix, "tocsr") else sisl_matrix
        matrices.append(csr.tocsr())
        if supercell_orders is not None:
            supercell_orders.append(supercell_order_from_sisl_matrix(sisl_matrix))
    return matrices
