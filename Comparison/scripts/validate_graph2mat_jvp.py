#!/usr/bin/env python3
"""C12 + C13 / E-F_001-S14: internal validation of the Graph2Mat directional JVP.

Three objects, one model, one geometry, one fixed neighbour topology and the
same direction ``v``:

``D_jvp[v]``
    one :func:`compute_graph2mat_directional_derivative` call -- the production
    path, which never materialises ``[n_outputs, N, 3]``.
``D_coord[v] = sum_(I,a) v_(I,a) D[e_(I,a)]``
    the legacy one-hot API contracted by hand, through the *public* selector
    (:func:`select_derivative_prediction_from_jacobian`) so that the cartesian
    frame convention of the two paths is compared, not assumed.
``D_frozen[v](delta) = [F(R + delta v) - F(R - delta v)] / (2 delta)``
    the model's own central difference, evaluated through the same forward that
    the JVP differentiates.

The two gates are deliberately different in kind:

* ``D_coord`` is the *same operator with different bookkeeping*. Its only
  admissible discrepancy is dtype roundoff (:data:`ROUNDOFF_TOLERANCE`); a
  frame, sign or ordering defect cannot hide under it.
* ``D_frozen`` is a *different numerical route*, so it can only agree inside the
  central-difference plateau. The amplitude is therefore swept and the residual
  must be positively explained -- noise-limited inside the plateau, or decaying
  as ``delta^2`` towards zero -- using the same classification policy as GO-2
  (:func:`certify_siesta_dhsdr.classify`).

"Small" is not a verdict. The legacy regression smoke tolerance
(:data:`SMOKE_TOLERANCE`, relative Frobenius 0.25) is carried in the report for
one purpose only: a residual that satisfies *nothing but* that is reported as a
GO-3 failure, with ``smoke_tolerance_only`` set.

Every JVP here is taken at fixed ``edge_index``/``shifts``, so a displacement
that moves a pair across the neighbour cutoff is a discontinuity of the graph
rather than a derivative: :func:`fd_perturbation_space.topology_margin` is
evaluated at every swept amplitude and GO-3 fails if any pair crosses. The
batch topology hash is re-read after every displaced forward, so a frozen
evaluation that silently rebuilt the graph fails too.

Errors are separated by dtype (a float32 model and a float64 model have
different roundoff floors and are never mixed into one number) and by backend.
CUDA is only *compared* when the cheap preflight of
:func:`resolve_jvp_backend` actually ran the double backward on this model;
every fallback to CPU is recorded with its reason instead of being silent.

Units: positions and amplitudes in Ang, derivatives in eV/Ang (the model's own
label units per Ang).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
from certify_siesta_dhsdr import check, classify  # noqa: E402
from graph2mat_autograd_derivatives import (  # noqa: E402
    CPU_BACKEND_RECORD,
    DEFAULT_OUTPUT_KEYS,
    batch_topology_hash,
    compute_graph2mat_directional_derivative,
    compute_graph2mat_position_jacobian,
    flatten_graph2mat_predictions,
    graph2mat_forward_labels,
    resolve_jvp_backend,
    select_derivative_prediction_from_jacobian,
)
from hamiltonian_derivative_stencil import (  # noqa: E402
    sparse_blockwise_hermiticity_defect,
)

SCHEMA = "graph2mat_jvp_internal_validation_v1"
GATE = "GO-3"

# The legacy regression tolerance of the real-checkpoint smoke test. It is a
# *refusal* threshold here: a residual that only satisfies this is unresolved.
SMOKE_TOLERANCE = 0.25

# D_jvp vs D_coord is the same operator evaluated twice; anything above the
# accumulated roundoff of ~3N contractions is a defect, not noise. ~1e3 * eps.
ROUNDOFF_TOLERANCE = {"float64": 1e-11, "float32": 1e-4}
# Uniform translation: relative to the largest directional response measured in
# the same run, so it is a statement about this model at this geometry.
TRANSLATION_TOLERANCE = {"float64": 1e-10, "float32": 1e-4}
# Same tensor computed on two devices: reductions are ordered differently, so
# the floor is the dtype's, not bitwise equality.
CROSS_BACKEND_TOLERANCE = {"float64": 1e-11, "float32": 1e-4}
# D(mu, nu, R) = D(nu, mu, -R) is exact algebra of a symmetric-matrix model.
HERMITICITY_TOLERANCE = {"float64": 1e-10, "float32": 1e-5}

LINEARITY_COEFFICIENTS = (0.37, -1.9)
# A response this far below the magnitude it was contracted from is a null
# direction (uniform translation), not a signal to take ratios against.
NULL_DIRECTION_FRACTION = 1e-9
# A central difference of an invariant quantity is pure cancellation noise,
# bounded by eps ||F|| / (2 delta). The margin is the slack over that bound.
FD_NOISE_MARGIN = 10.0


class JvpValidationError(RuntimeError):
    """The validation cannot be evaluated at all (unusable inputs)."""


def _tolerance(table: Mapping[str, float], dtype_name: str, what: str) -> float:
    try:
        return float(table[dtype_name])
    except KeyError as exc:
        raise JvpValidationError(
            f"No pre-registered {what} tolerance for dtype {dtype_name!r}; add one "
            "explicitly instead of falling back to a looser dtype."
        ) from exc


# --------------------------------------------------------------------------- #
# The three paths
# --------------------------------------------------------------------------- #


def direction_vectors(direction: Any) -> np.ndarray:
    """``[N, 3]`` float64 vectors of an ``fdp.Direction`` or a raw array."""
    return np.asarray(getattr(direction, "vectors", direction), dtype=np.float64)


def _flat64(predictions: Mapping[str, torch.Tensor], output_keys: Sequence[str]) -> torch.Tensor:
    flat, _ = flatten_graph2mat_predictions(dict(predictions), output_keys=output_keys)
    return flat.detach().to(torch.float64).cpu()


def _batch_tangent(
    vectors: np.ndarray, positions: torch.Tensor, change_of_basis: Any | None
) -> torch.Tensor:
    """Cartesian ``v`` expressed in the frame the batch positions live in.

    ``p_batch = C @ p_cart``, so a cartesian displacement maps to ``C v`` per
    atom. Identical convention to the JVP and to
    :func:`select_derivative_prediction_from_jacobian`; that it *is* identical
    is what the coordinate-combination check measures.
    """
    tangent = torch.as_tensor(vectors, dtype=positions.dtype, device=positions.device)
    if change_of_basis is None:
        return tangent
    cob = torch.as_tensor(change_of_basis, dtype=positions.dtype, device=positions.device)
    return tangent @ cob.T


def directional_jvp(
    model: torch.nn.Module,
    batch: Any,
    direction: Any,
    *,
    change_of_basis: Any | None = None,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
    data_processor: Any | None = None,
    threshold: float | None = None,
    backend: Any = CPU_BACKEND_RECORD,
) -> tuple[torch.Tensor, Any]:
    """``D_jvp[v]`` flattened to float64, plus the full production result."""
    result = compute_graph2mat_directional_derivative(
        model,
        batch,
        direction,
        change_of_basis=change_of_basis,
        output_keys=output_keys,
        data_processor=data_processor,
        threshold=threshold,
        backend=backend,
    )
    return _flat64(result.derivative, output_keys), result


def coordinate_combination(
    model: torch.nn.Module,
    batch: Any,
    direction: Any,
    *,
    change_of_basis: Any | None = None,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
    jacobian: Any | None = None,
) -> tuple[torch.Tensor, float]:
    """``sum_(I,a) v_(I,a) D[e_(I,a)]`` through the public one-hot selector.

    Deliberately the expensive route: it materialises the full
    ``[n_outputs, N, 3]`` jacobian, which is exactly what the directional API
    exists to avoid. It is a control, only ever run on validation-sized systems.

    Returns the contraction and ``||J||_F``, the scale the contracted terms came
    from: for a direction the operator annihilates (a uniform translation) the
    result is zero by cancellation, and the residual of that cancellation can
    only be judged against the magnitude that cancelled.
    """
    vectors = direction_vectors(direction)
    if jacobian is None:
        jacobian = compute_graph2mat_position_jacobian(
            model, batch, method="jvp_double_backward", output_keys=output_keys
        )
    if vectors.shape[0] != jacobian.n_atoms:
        raise JvpValidationError(
            f"direction spans {vectors.shape[0]} atoms, structure has {jacobian.n_atoms}."
        )
    total: torch.Tensor | None = None
    for atom in range(vectors.shape[0]):
        for axis in range(3):
            weight = float(vectors[atom, axis])
            if weight == 0.0:
                continue
            column = _flat64(
                select_derivative_prediction_from_jacobian(
                    jacobian.jacobian,
                    jacobian.spec,
                    atom,
                    axis,
                    change_of_basis=change_of_basis,
                ),
                output_keys,
            )
            total = weight * column if total is None else total + weight * column
    if total is None:
        raise JvpValidationError("direction is identically zero; nothing to contract.")
    return total, float(torch.linalg.norm(jacobian.jacobian.detach().to(torch.float64)))


def frozen_central_difference(
    model: torch.nn.Module,
    batch: Any,
    direction: Any,
    delta_ang: float,
    *,
    change_of_basis: Any | None = None,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
) -> tuple[torch.Tensor, list[str]]:
    """``[F(R + d v) - F(R - d v)] / (2 d)`` at frozen topology.

    Uses :func:`graph2mat_forward_labels`, i.e. the very forward the JVP
    differentiates, with only ``positions`` substituted: ``edge_index`` and
    ``shifts`` are untouched. The topology hash of each displaced batch is
    returned so the caller can prove the graph did not move.
    """
    delta = float(delta_ang)
    if delta <= 0.0:
        raise JvpValidationError("delta_ang must be positive.")
    positions = batch["positions"]
    tangent = _batch_tangent(direction_vectors(direction), positions, change_of_basis)
    evaluations: dict[float, torch.Tensor] = {}
    hashes: list[str] = []
    for sign in fdp.signs_for_method("central"):
        displaced = positions.detach().clone() + float(sign) * delta * tangent
        flat, spec = graph2mat_forward_labels(model, batch, displaced, output_keys=output_keys)
        evaluations[float(sign)] = flat.detach().to(torch.float64).cpu()
        hashes.append(batch_topology_hash(batch)["topology_hash"])
    return (evaluations[1.0] - evaluations[-1.0]) / (2.0 * delta), hashes


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def discrepancy(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    scale: float | None = None,
    null_fraction: float = NULL_DIRECTION_FRACTION,
) -> dict[str, float]:
    """Absolute and relative agreement of two flat derivative vectors.

    ``scale`` is the magnitude the two vectors were built from. When both
    responses are negligible against it -- a direction the operator annihilates,
    where "relative to the response" would divide two roundoff residuals by each
    other -- the residual is reported relative to ``scale`` instead, and
    ``is_null_direction`` says so.
    """
    if reference.shape != candidate.shape:
        raise JvpValidationError(
            f"cannot compare derivatives of shapes {tuple(reference.shape)} and "
            f"{tuple(candidate.shape)}."
        )
    diff_norm = float(torch.linalg.norm(reference - candidate))
    max_abs = float((reference - candidate).abs().max()) if reference.numel() else 0.0
    reference_norm = float(torch.linalg.norm(reference))
    candidate_norm = float(torch.linalg.norm(candidate))
    signal = max(reference_norm, candidate_norm)
    is_null = scale is not None and signal <= float(null_fraction) * float(scale)
    denominator = float(scale) if is_null else signal
    cosine = (
        float(torch.dot(reference, candidate)) / (reference_norm * candidate_norm)
        if reference_norm > 0.0 and candidate_norm > 0.0
        else float("nan")
    )
    return {
        "frobenius": diff_norm,
        "max_abs": max_abs,
        "relative_frobenius": diff_norm / denominator if denominator > 0.0 else 0.0,
        "cosine": cosine,
        "reference_frobenius": reference_norm,
        "candidate_frobenius": candidate_norm,
        "is_null_direction": bool(is_null),
        "relative_to": "cancellation_scale" if is_null else "response",
    }


def noise_floor(frozen_by_delta: Mapping[float, torch.Tensor], deltas: Sequence[float]) -> float:
    """``tau``: the largest disagreement among plateau amplitudes.

    Every plateau amplitude estimates the same derivative, so their mutual
    spread is what this finite difference can resolve at all.
    """
    ordered = sorted(float(delta) for delta in deltas)
    tau = 0.0
    for first in range(len(ordered)):
        for second in range(first + 1, len(ordered)):
            tau = max(
                tau,
                float(
                    torch.linalg.norm(
                        frozen_by_delta[ordered[first]] - frozen_by_delta[ordered[second]]
                    )
                ),
            )
    return tau


def frozen_plateau(frozen_by_delta: Mapping[float, torch.Tensor]) -> dict[str, Any]:
    """Amplitudes over which the frozen central difference is flat, or why not."""
    measurements = {
        float(delta): float(torch.linalg.norm(value))
        for delta, value in frozen_by_delta.items()
    }
    try:
        selection = fdp.select_delta_from_plateau(measurements)
    except fdp.FdPerturbationError as exc:
        return {"has_plateau": False, "reason": str(exc), "frozen_frobenius_by_delta": measurements}
    return {"has_plateau": True, "frozen_frobenius_by_delta": measurements, **selection}


# --------------------------------------------------------------------------- #
# Per-direction checks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationInputs:
    """Everything that does not change from direction to direction."""

    change_of_basis: Any | None = None
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS
    data_processor: Any | None = None
    threshold: float | None = None
    cutoff_ang: float | Sequence[float] | None = None
    positions_ang: Any | None = None
    lattice_vectors_ang: Any | None = None


def cartesian_positions(batch: Any, change_of_basis: Any | None) -> np.ndarray:
    """Batch positions back in the physical cartesian frame (``p_cart = C^-1 p_batch``)."""
    positions = batch["positions"].detach().to(torch.float64).cpu().numpy()
    if change_of_basis is None:
        return positions
    cob = np.asarray(change_of_basis, dtype=np.float64)
    return positions @ np.linalg.inv(cob).T


def _topology_check(
    direction: Any,
    deltas: Sequence[float],
    inputs: ValidationInputs,
    batch: Any,
    context: str,
) -> dict[str, Any]:
    """Neighbour-list stability of ``R +- delta v`` at every swept amplitude."""
    if inputs.cutoff_ang is None:
        return check(
            "topology_fixed",
            False,
            "no neighbour cutoff was supplied, so a fixed-topology derivative cannot be "
            "certified; pass cutoff_ang (Graph2Mat edge cutoff or per-atom PAO radii)",
            direction_name=context,
            status="not_evaluable",
        )
    positions = (
        np.asarray(inputs.positions_ang, dtype=np.float64)
        if inputs.positions_ang is not None
        else cartesian_positions(batch, inputs.change_of_basis)
    )
    margins = []
    for delta in deltas:
        margin = fdp.topology_margin(
            positions,
            direction=direction,
            delta_ang=float(delta),
            cutoff_ang=inputs.cutoff_ang,
            lattice_vectors_ang=inputs.lattice_vectors_ang,
        )
        margins.append({"delta_ang": float(delta), **margin.to_dict()})
    preserved = all(entry["topology_preserved"] for entry in margins)
    worst = min((entry["min_margin_ang"] for entry in margins), default=float("inf"))
    return check(
        "topology_fixed",
        preserved,
        f"neighbour list of R +- delta v is unchanged at every swept amplitude "
        f"(smallest margin {worst:.4g} Ang)"
        if preserved
        else "a pair crosses the neighbour cutoff: a derivative across a graph "
        "discontinuity is not a derivative",
        direction_name=context,
        min_margin_ang=worst,
        margins=margins,
    )


def validate_direction(
    model: torch.nn.Module,
    batch: Any,
    direction: Any,
    *,
    deltas: Sequence[float],
    dtype_name: str,
    inputs: ValidationInputs,
    backend: Any = CPU_BACKEND_RECORD,
    jacobian: Any | None = None,
) -> dict[str, Any]:
    """All GO-3 checks for one direction: coordinate combination, frozen, topology.

    ``jacobian`` is the full ``[n_outputs, N, 3]`` control jacobian; it does not
    depend on the direction, so the caller computes it once per run.
    """
    name = str(getattr(direction, "name", "direction"))
    kind = str(getattr(direction, "kind", fdp.KIND_COLLECTIVE))
    is_translation = kind == fdp.KIND_UNIFORM_TRANSLATION
    base_topology = batch_topology_hash(batch)

    jvp_flat, result = directional_jvp(
        model,
        batch,
        direction,
        change_of_basis=inputs.change_of_basis,
        output_keys=inputs.output_keys,
        data_processor=inputs.data_processor,
        threshold=inputs.threshold,
        backend=backend,
    )
    coordinate_flat, jacobian_frobenius = coordinate_combination(
        model,
        batch,
        direction,
        change_of_basis=inputs.change_of_basis,
        output_keys=inputs.output_keys,
        jacobian=jacobian,
    )
    roundoff = _tolerance(ROUNDOFF_TOLERANCE, dtype_name, "coordinate-combination")
    against_coordinates = discrepancy(
        coordinate_flat,
        jvp_flat,
        # A direction the operator annihilates to within this dtype's roundoff
        # is judged against the magnitude that cancelled, not against zero.
        scale=jacobian_frobenius * float(np.linalg.norm(direction_vectors(direction))),
        null_fraction=roundoff,
    )
    checks = [
        check(
            "coordinate_combination",
            against_coordinates["relative_frobenius"] <= roundoff,
            f"D_jvp[v] vs sum_(I,a) v_(I,a) D[e_(I,a)]: relative Frobenius "
            f"{against_coordinates['relative_frobenius']:.3e} vs {dtype_name} roundoff "
            f"tolerance {roundoff:.3e}",
            direction_name=name,
            dtype=dtype_name,
            **against_coordinates,
        )
    ]

    frozen_by_delta: dict[float, torch.Tensor] = {}
    topology_hashes = [base_topology["topology_hash"]]
    for delta in deltas:
        frozen, hashes = frozen_central_difference(
            model,
            batch,
            direction,
            delta,
            change_of_basis=inputs.change_of_basis,
            output_keys=inputs.output_keys,
        )
        frozen_by_delta[float(delta)] = frozen
        topology_hashes.extend(hashes)

    checks.append(
        check(
            "batch_topology_unchanged",
            len(set(topology_hashes)) == 1,
            "the JVP and every displaced frozen forward ran on the same graph "
            f"({base_topology['topology_hash'][:12]}...)",
            direction_name=name,
            dtype=dtype_name,
            topology_hashes=sorted(set(topology_hashes)),
            **base_topology,
        )
    )
    checks.append(_topology_check(direction, deltas, inputs, batch, name))

    plateau = frozen_plateau(frozen_by_delta)
    frozen_rows = []
    for delta in sorted(frozen_by_delta):
        frozen_rows.append(
            {
                "delta_ang": delta,
                "direction_name": name,
                "dtype": dtype_name,
                **discrepancy(frozen_by_delta[delta], jvp_flat),
            }
        )
    best_relative = min((row["relative_frobenius"] for row in frozen_rows), default=float("nan"))

    if is_translation:
        # A uniform translation of a fixed-shift graph changes no relative
        # vector, so both routes must return zero; a plateau of zeros is not a
        # plateau, which is why this direction is gated on magnitude instead.
        frozen_checks = [
            _translation_check(
                jvp_flat,
                frozen_by_delta,
                name,
                dtype_name,
                float(torch.linalg.norm(_flat64(result.base_predictions, inputs.output_keys))),
            )
        ]
        classification = None
    elif not plateau["has_plateau"]:
        classification = None
        frozen_checks = [
            check(
                "jvp_vs_frozen",
                False,
                "the frozen central difference never stabilises over the swept amplitudes, "
                f"so there is no plateau to compare the JVP against: {plateau['reason']}",
                direction_name=name,
                dtype=dtype_name,
                **plateau,
            )
        ]
    else:
        plateau_deltas = [float(value) for value in plateau["plateau_delta_ang"]]
        tau = noise_floor(frozen_by_delta, plateau_deltas)
        classification = classify(
            deltas=plateau_deltas,
            discrepancy_by_delta={
                delta: float(torch.linalg.norm(frozen_by_delta[delta] - jvp_flat))
                for delta in plateau_deltas
            },
            tau=tau,
            is_null_direction=False,
        )
        smoke_only = bool(
            not classification["passed"] and best_relative <= SMOKE_TOLERANCE
        )
        frozen_checks = [
            check(
                "jvp_vs_frozen",
                bool(classification["passed"]),
                f"JVP vs frozen central difference inside the plateau {plateau_deltas}: "
                f"{classification['reason']}",
                direction_name=name,
                dtype=dtype_name,
                smoke_tolerance=SMOKE_TOLERANCE,
                smoke_tolerance_only=smoke_only,
                best_relative_frobenius=best_relative,
                **plateau,
                # 'passed'/'reason' are the check's own fields; keep the rest.
                **{
                    key: value
                    for key, value in classification.items()
                    if key not in ("passed", "reason")
                },
            )
        ]

    hermiticity = _hermiticity_check(result, name, dtype_name, inputs)
    if hermiticity is not None:
        checks.append(hermiticity)
    checks.extend(frozen_checks)

    return {
        "direction_name": name,
        "direction_kind": kind,
        "dtype": dtype_name,
        "direction_hash": result.metadata.get("direction_hash"),
        "jvp_metadata": result.metadata,
        "jvp_frobenius": float(torch.linalg.norm(jvp_flat)),
        "coordinate_combination": against_coordinates,
        "frozen_by_delta": frozen_rows,
        "plateau": plateau,
        "classification": classification,
        "checks": checks,
        "flat": jvp_flat,
    }


def _translation_check(
    jvp_flat: torch.Tensor,
    frozen_by_delta: Mapping[float, torch.Tensor],
    name: str,
    dtype_name: str,
    base_norm: float,
) -> dict[str, Any]:
    """A uniform translation must leave the labels invariant on both routes.

    The two routes are held to different floors on purpose. The JVP is an
    analytic zero, so it is judged against the run's own signal. The frozen
    central difference of an invariant quantity is *pure cancellation noise*:
    it cannot be smaller than ``eps ||F|| / (2 delta)``, which grows as the
    amplitude shrinks, so holding it to a fixed relative number would only
    measure the smallest amplitude in the sweep.
    """
    tolerance = _tolerance(TRANSLATION_TOLERANCE, dtype_name, "translation")
    eps = float(torch.finfo(getattr(torch, dtype_name)).eps)
    jvp_norm = float(torch.linalg.norm(jvp_flat))
    frozen_rows = []
    for delta in sorted(frozen_by_delta):
        floor = FD_NOISE_MARGIN * eps * base_norm / (2.0 * float(delta))
        measured = float(torch.linalg.norm(frozen_by_delta[delta]))
        frozen_rows.append(
            {
                "delta_ang": float(delta),
                "frozen_frobenius": measured,
                "fd_noise_floor": floor,
                "within_fd_noise": measured <= floor,
            }
        )
    frozen_ok = all(row["within_fd_noise"] for row in frozen_rows)
    worst = max((row["frozen_frobenius"] for row in frozen_rows), default=0.0)
    return check(
        "uniform_translation_invariance",
        frozen_ok,  # the JVP half needs the run scale and is resolved by the caller
        f"|D[t]| = {jvp_norm:.3e} (JVP), {worst:.3e} (frozen, worst amplitude; "
        f"{'at' if frozen_ok else 'above'} the {dtype_name} FD noise floor)",
        direction_name=name,
        dtype=dtype_name,
        jvp_frobenius=jvp_norm,
        frozen_frobenius_max=worst,
        frozen_by_delta=frozen_rows,
        base_label_frobenius=base_norm,
        tolerance=tolerance,
        status="pending_run_scale",
    )


def _hermiticity_check(
    result: Any, name: str, dtype_name: str, inputs: ValidationInputs
) -> dict[str, Any] | None:
    """Real-space blockwise hermiticity of the serialised ``D_H[v]``."""
    if inputs.data_processor is None:
        return None
    tolerance = _tolerance(HERMITICITY_TOLERANCE, dtype_name, "hermiticity")
    if not result.matrices or not result.supercell_orders or not result.supercell_orders[0]:
        return check(
            "blockwise_hermiticity",
            False,
            "a data_processor was supplied but the directional derivative produced no "
            "sparse matrix/supercell order to test D(mu, nu, R) = D(nu, mu, -R) on",
            direction_name=name,
            dtype=dtype_name,
            status="not_evaluable",
        )
    defect = sparse_blockwise_hermiticity_defect(result.matrices[0], result.supercell_orders[0])
    finite = bool(np.isfinite(defect))
    return check(
        "blockwise_hermiticity",
        finite and defect <= tolerance,
        f"D(mu, nu, R) vs D(nu, mu, -R): relative defect {defect:.3e} vs tolerance "
        f"{tolerance:.3e}"
        if finite
        else "hermiticity is not computable on this supercell layout (no -R partners)",
        direction_name=name,
        dtype=dtype_name,
        hermiticity_defect=float(defect),
        tolerance=tolerance,
    )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def _cast(model: torch.nn.Module, batch: Any, dtype: torch.dtype) -> tuple[torch.nn.Module, Any]:
    """Model and batch in one dtype, leaving integer fields (topology) alone.

    ``batch.to(dtype)`` would cast ``edge_index`` too, so only floating tensors
    are converted.
    """
    batch = batch.clone() if hasattr(batch, "clone") else dict(batch)
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            batch[key] = value.to(dtype)
    return model.to(dtype), batch


def _linearity_check(
    model: torch.nn.Module,
    batch: Any,
    directions: Sequence[Any],
    *,
    dtype_name: str,
    inputs: ValidationInputs,
    backend: Any,
) -> dict[str, Any]:
    """``D[a v + b w] = a D[v] + b D[w]``: the operator is linear or it is not a JVP."""
    if len(directions) < 2:
        return check(
            "linearity",
            False,
            "linearity needs two independent directions; the direction set has "
            f"{len(directions)}",
            dtype=dtype_name,
            status="not_evaluable",
        )
    left, right = directions[0], directions[1]
    a, b = LINEARITY_COEFFICIENTS
    combined = a * direction_vectors(left) + b * direction_vectors(right)
    flats = [
        directional_jvp(
            model,
            batch,
            vectors,
            change_of_basis=inputs.change_of_basis,
            output_keys=inputs.output_keys,
            backend=backend,
        )[0]
        for vectors in (left, right, combined)
    ]
    measured = discrepancy(a * flats[0] + b * flats[1], flats[2])
    tolerance = _tolerance(ROUNDOFF_TOLERANCE, dtype_name, "linearity")
    return check(
        "linearity",
        measured["relative_frobenius"] <= tolerance,
        f"D[{a} v + {b} w] vs {a} D[v] + {b} D[w] on "
        f"({getattr(left, 'name', 'v')}, {getattr(right, 'name', 'w')}): relative Frobenius "
        f"{measured['relative_frobenius']:.3e} vs {dtype_name} roundoff tolerance "
        f"{tolerance:.3e}",
        dtype=dtype_name,
        coefficients=list(LINEARITY_COEFFICIENTS),
        **measured,
    )


def validate_directional_jvp(
    model: torch.nn.Module,
    batch: Any,
    *,
    directions: Sequence[Any] | None = None,
    deltas: Sequence[float] | None = None,
    dtypes: Sequence[str] = ("float64",),
    backends: Sequence[str] = ("cpu",),
    change_of_basis: Any | None = None,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
    data_processor: Any | None = None,
    threshold: float | None = None,
    cutoff_ang: float | Sequence[float] | None = None,
    positions_ang: Any | None = None,
    lattice_vectors_ang: Any | None = None,
) -> dict[str, Any]:
    """Run the whole GO-3 internal validation and return the report.

    ``directions`` defaults to the S10 minimum (one cartesian, one random
    collective, one uniform translation) plus a second collective, because
    linearity needs two independent ones. ``deltas`` defaults to a five-point
    geometric sweep: three is the minimum for a plateau, five lets the residual
    exponent be fitted rather than assumed.
    """

    n_atoms = int(batch["positions"].shape[0])
    if directions is None:
        directions = [
            *fdp.default_direction_set(n_atoms, seed=0),
            fdp.random_collective(n_atoms, seed=1),
        ]
    if deltas is None:
        deltas = fdp.delta_sweep(count=5)
    deltas = sorted(float(delta) for delta in deltas)
    sweep_status = fdp.delta_sweep_status(deltas)
    inputs = ValidationInputs(
        change_of_basis=change_of_basis,
        output_keys=tuple(output_keys),
        data_processor=data_processor,
        threshold=threshold,
        cutoff_ang=cutoff_ang,
        positions_ang=positions_ang,
        lattice_vectors_ang=lattice_vectors_ang,
    )

    checks: list[dict[str, Any]] = [
        check(
            "delta_sweep_plateau_capable",
            bool(sweep_status["delta_plateau_ready"]),
            f"{sweep_status['delta_count']} amplitudes swept "
            f"({sweep_status['delta_sweep_status']})",
            **sweep_status,
        )
    ]

    runs: list[dict[str, Any]] = []
    flats_by_key: dict[tuple[str, str, str], torch.Tensor] = {}
    for dtype_name in dtypes:
        try:
            torch_dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise JvpValidationError(f"Unknown dtype {dtype_name!r}.") from exc
        for requested_backend in backends:
            cast_model, cast_batch = _cast(model, batch, torch_dtype)
            run_model, run_batch, record = resolve_jvp_backend(
                cast_model, cast_batch, requested_backend, output_keys=inputs.output_keys
            )
            # The control jacobian is direction-independent: 3N JVPs once per
            # run, not once per direction.
            jacobian = compute_graph2mat_position_jacobian(
                run_model, run_batch, method="jvp_double_backward", output_keys=inputs.output_keys
            )
            direction_results = [
                validate_direction(
                    run_model,
                    run_batch,
                    direction,
                    deltas=deltas,
                    dtype_name=dtype_name,
                    inputs=inputs,
                    backend=record,
                    jacobian=jacobian,
                )
                for direction in directions
            ]
            collective_scale = max(
                (
                    entry["jvp_frobenius"]
                    for entry in direction_results
                    if entry["direction_kind"] != fdp.KIND_UNIFORM_TRANSLATION
                ),
                default=0.0,
            )
            run_checks: list[dict[str, Any]] = []
            for entry in direction_results:
                for row in entry["checks"]:
                    if row.get("status") == "pending_run_scale":
                        row = _resolve_translation_verdict(row, collective_scale)
                    run_checks.append({**row, "backend": record.effective})
            run_checks.append(
                {
                    **_linearity_check(
                        run_model,
                        run_batch,
                        directions,
                        dtype_name=dtype_name,
                        inputs=inputs,
                        backend=record,
                    ),
                    "backend": record.effective,
                }
            )
            run_checks.append(
                {
                    **check(
                        "backend_preflight_documented",
                        record.preflight != "failed" or bool(record.reason),
                        f"requested {record.requested}, ran on {record.effective} "
                        f"(preflight {record.preflight})"
                        + (f"; fallback reason: {record.reason}" if record.reason else ""),
                        dtype=dtype_name,
                        **record.to_metadata(),
                    ),
                    "backend": record.effective,
                }
            )
            checks.extend(run_checks)
            for entry in direction_results:
                flats_by_key[(dtype_name, record.effective, entry["direction_name"])] = entry.pop(
                    "flat"
                )
            runs.append(
                {
                    "dtype": dtype_name,
                    "requested_backend": record.requested,
                    "effective_backend": record.effective,
                    "backend_preflight": record.preflight,
                    "backend_fallback_reason": record.reason,
                    "collective_signal_frobenius": collective_scale,
                    "directions": direction_results,
                }
            )

    checks.extend(_cross_backend_checks(flats_by_key))
    checks.append(_coverage_check(runs, deltas))

    failed = [row["check"] for row in checks if not row["passed"]]
    return {
        "schema": SCHEMA,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_atoms": n_atoms,
        "delta_ang_values": deltas,
        "direction_names": [str(getattr(d, "name", "direction")) for d in directions],
        "dtypes": list(dtypes),
        "requested_backends": list(backends),
        "tolerances": {
            "roundoff_relative_frobenius": ROUNDOFF_TOLERANCE,
            "uniform_translation_relative": TRANSLATION_TOLERANCE,
            "cross_backend_relative_frobenius": CROSS_BACKEND_TOLERANCE,
            "blockwise_hermiticity": HERMITICITY_TOLERANCE,
            "refused_smoke_tolerance": SMOKE_TOLERANCE,
        },
        "runs": runs,
        "checks": checks,
        "summary": {
            "passed": not failed,
            "gate": GATE,
            "checks_total": len(checks),
            "checks_failed": failed,
        },
    }


def _resolve_translation_verdict(row: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """Add the JVP half of the verdict, which needs the run's own signal scale."""
    tolerance = float(row["tolerance"])
    relative = float(row["jvp_frobenius"]) / scale if scale > 0.0 else float("inf")
    return {
        **row,
        "passed": bool(row["passed"]) and relative <= tolerance,
        "status": "evaluated",
        "collective_signal_frobenius": scale,
        "relative_to_signal": relative,
        "detail": f"{row['detail']}; the JVP is {relative:.3e} of the collective signal "
        f"{scale:.3e} vs tolerance {tolerance:.3e}",
    }


def _cross_backend_checks(
    flats_by_key: Mapping[tuple[str, str, str], torch.Tensor]
) -> list[dict[str, Any]]:
    """CPU vs CUDA, and only where CUDA actually ran the same direction and dtype."""
    rows: list[dict[str, Any]] = []
    for (dtype_name, backend, name), flat in sorted(flats_by_key.items()):
        if backend == "cpu":
            continue
        reference = flats_by_key.get((dtype_name, "cpu", name))
        if reference is None:
            rows.append(
                check(
                    "backend_equivalence",
                    True,
                    f"{backend} ran without a cpu counterpart for {name}/{dtype_name}; "
                    "no equivalence was claimed",
                    dtype=dtype_name,
                    backend=backend,
                    direction_name=name,
                    status="not_compared",
                )
            )
            continue
        measured = discrepancy(reference, flat)
        tolerance = _tolerance(CROSS_BACKEND_TOLERANCE, dtype_name, "cross-backend")
        rows.append(
            check(
                "backend_equivalence",
                measured["relative_frobenius"] <= tolerance,
                f"cpu vs {backend} on {name}/{dtype_name}: relative Frobenius "
                f"{measured['relative_frobenius']:.3e} vs tolerance {tolerance:.3e}",
                dtype=dtype_name,
                backend=backend,
                direction_name=name,
                status="compared",
                **measured,
            )
        )
    return rows


def _coverage_check(runs: Sequence[Mapping[str, Any]], deltas: Sequence[float]) -> dict[str, Any]:
    """An empty or one-sided validation must not be able to pass GO-3."""
    collective = [
        entry
        for run in runs
        for entry in run["directions"]
        if entry["direction_kind"] != fdp.KIND_UNIFORM_TRANSLATION
    ]
    classified = [entry for entry in collective if entry["classification"] is not None]
    translations = [
        entry
        for run in runs
        for entry in run["directions"]
        if entry["direction_kind"] == fdp.KIND_UNIFORM_TRANSLATION
    ]
    enough = bool(runs) and bool(classified) and bool(translations) and len(deltas) >= 3
    return check(
        "coverage",
        enough,
        f"{len(runs)} run(s), {len(collective)} collective direction(s) of which "
        f"{len(classified)} compared against a frozen plateau, {len(translations)} uniform "
        f"translation(s), {len(deltas)} amplitudes",
        runs=len(runs),
        collective_directions=len(collective),
        plateau_classified_directions=len(classified),
        translation_directions=len(translations),
        delta_count=len(deltas),
    )


def demo() -> None:
    """Self-check on a translation-invariant analytic model (no checkpoint needed)."""

    class PairModel(torch.nn.Module):
        """Smooth, translation-invariant function of the interatomic vectors."""

        def forward(self, data):
            positions = data["positions"]
            delta = positions[:, None, :] - positions[None, :, :]
            distance_sq = delta.pow(2).sum(-1)
            return {
                "node_labels": torch.exp(-distance_sq).sum(1),
                "edge_labels": torch.sin(delta).reshape(-1),
            }

    class Batch(dict):
        def clone(self):
            return Batch(self)

    torch.manual_seed(0)
    batch = Batch(positions=torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [0.7, 1.2, 0.1]]).double())
    report = validate_directional_jvp(
        PairModel(),
        batch,
        cutoff_ang=6.0,
        deltas=fdp.delta_sweep(0.004, count=5),
    )
    assert report["summary"]["passed"], report["summary"]["checks_failed"]
    print(f"validate_graph2mat_jvp: {GATE} ok ({report['summary']['checks_total']} checks)")


if __name__ == "__main__":
    demo()
