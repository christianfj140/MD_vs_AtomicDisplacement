#!/usr/bin/env python3
"""C16 / E-F_001-S19: gauge-invariant subspace metrics in the S metric (GO-6).

C15 persists ``C`` with ``C†SC = I`` per k. This module is what may be *said*
about those windows once a degeneracy makes the band index meaningless. Two
S-orthonormal windows are compared through

    M = C_A† S C_B,        sigma_i(M) = cos(theta_i)

with ``theta_i`` the principal angles between the two spans. Everything else is
a function of ``M``:

    projector distance   ||P_A - P_B||_S = sqrt(d_A + d_B - 2*sum sigma_i^2)

because ``tr(P_A P_B) = ||M||_F^2`` for the S-orthogonal projectors
``P = C C† S``. Those are S-self-adjoint rather than Hermitian, so the norm is
the Frobenius norm in the S metric, ``||S^{1/2}(P_A-P_B)S^{-1/2}||_F``. No
``no_u x no_u`` projector is ever built, which is what makes this usable at
MATBG size.

Three things this module refuses to do:

* cluster by band index. :func:`near_degenerate_clusters` splits on the gap
  only where the gap is *resolved*, and the resolution comes from the persisted
  eigen-residual and the window conditioning (:func:`eigenvalue_resolutions_ev`),
  not from a hand-picked eV;
* report a tolerance it did not derive. ``C†SC`` is only ``I`` to the identity
  error C15 measured, and that error propagates into the singular values, the
  angles and the projector distance with different powers
  (:func:`subspace_metric_tolerances`);
* call a band-indexed number gauge invariant. Under ``C -> C U`` with ``U``
  block diagonal in the clusters, each cluster's span is fixed and a coupling
  block goes to ``U† g U``; :func:`block_metrics` publishes only the singular
  values and Frobenius norms of :mod:`epc_formalism`.

The gate this certifies is GO-6, which the roadmap requires *before* graphene K
or any TBG validation: :func:`require_go6` is how a downstream script asks.

Units: energies, resolutions and gaps in eV; angles in radians; the singular
values and the projector distance are dimensionless.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from epc_formalism import gauge_invariant_block_metrics  # noqa: E402
from run_deeph_sparse_spectrum import (  # noqa: E402
    DEFAULT_EIGENSPACE_DEGENERACY_EV,
    EIGENSPACE_IDENTITY_TOLERANCE_POLICY,
    eigenspace_identity_tolerance,
    load_persisted_eigenspace,
    persisted_eigenspace_data,
)

GATE_ID = "go6_subspace_metrics_v1"
SCHEMA = "epc_subspace_certification_v1"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/subspaces"
REPORT_NAME = "go6_subspace_metrics.json"

# How an eigenvalue's uncertainty is read off the persisted diagnostics. With
# ``c`` S-normalised and ``r = H c - eps S c``, the nearest exact eigenvalue of
# the pencil is within ``||r||_{S^-1} <= ||r||_2 / sqrt(lambda_min)``. The solver
# persists the *relative* residual ``rho = ||r|| / (||Hc|| + |eps| ||Sc||)``, and
# at convergence ``||Hc|| ~ |eps| ||Sc||``, so ``||r|| ~ 2 rho |eps| ||Sc||`` with
# ``||Sc||_2 <= sqrt(lambda_max)``; the two metric eigenvalues combine into
# ``sqrt(cond)``. ``|eps|`` alone would understate the scale whenever the solver
# measures energies from E_F (graphene's Dirac point sits at 0), so the window
# span is the floor of the energy scale. The dimensionless factor is C15's own
# identity budget, so the two tolerance policies cannot drift apart.
ENERGY_RESOLUTION_POLICY = (
    "eta_n = 2 * sqrt(cond(V^dag S V)) * max(|eps_n|, span(window)) * "
    f"[{EIGENSPACE_IDENTITY_TOLERANCE_POLICY}]"
)

# ``C†SC = I + D`` with ``|D| <= delta``: Weyl moves every singular value of M by
# at most ``delta_A + delta_B``. The angle is *not* Lipschitz there — near
# ``sigma = 1``, ``cos(theta) = 1 - d`` gives ``theta ~ sqrt(2 d)`` — and the
# projector distance inherits ``sqrt(4 d_min * delta)`` from
# ``d_A + d_B - 2 sum sigma^2``. Same measured error, three different powers.
METRIC_TOLERANCE_POLICY = (
    "delta = max(identity_error_A, no_u*eps_f64) + max(identity_error_B, no_u*eps_f64); "
    "singular value <= delta; principal angle <= sqrt(2*delta); "
    "projector Frobenius distance <= sqrt(4*min(d_A,d_B)*delta)"
)


class SubspaceGateError(RuntimeError):
    """A subspace claim was made that GO-6 does not support."""


# --------------------------------------------------------------------------- #
# Resolution-driven clustering
# --------------------------------------------------------------------------- #


def eigenvalue_resolutions_ev(
    energies: Any,
    relative_residuals: Any,
    condition_number: float,
    orbital_count: int,
) -> np.ndarray:
    """Per-state energy resolution in eV, by :data:`ENERGY_RESOLUTION_POLICY`.

    ``relative_residuals`` is the solver's ``generalized_relative_residual``,
    either per state or one conservative maximum for the whole window.
    """
    values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0)
    residuals = np.broadcast_to(
        np.asarray(relative_residuals, dtype=np.float64), values.shape
    )
    condition = max(float(condition_number), 1.0)
    span = float(values.max() - values.min())
    scale = np.maximum(np.abs(values), span)
    dimensionless = np.array(
        [eigenspace_identity_tolerance(float(rho), condition, orbital_count) for rho in residuals]
    )
    return 2.0 * np.sqrt(condition) * scale * dimensionless


def near_degenerate_clusters(
    energies: Any, resolutions: Any, *, separation_factor: float = 1.0
) -> list[int]:
    """One cluster label per state: neighbours split only where the gap is resolved.

    Single linkage on the *measured* gaps — a chain of individually unresolved
    gaps is one cluster, which is the honest reading of a window the solver
    cannot separate. With a constant resolution this reproduces
    ``run_deeph_sparse_spectrum.degenerate_subspace_labels`` exactly.
    """
    values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []
    if np.any(np.diff(values) < -1e-12):
        raise SubspaceGateError("Eigenvalues are not ascending; cannot cluster by gap")
    limits = np.broadcast_to(np.asarray(resolutions, dtype=np.float64), values.shape)
    threshold = separation_factor * np.maximum(limits[:-1], limits[1:])
    return np.concatenate(([0], np.cumsum(np.diff(values) > threshold))).astype(int).tolist()


def cluster_groups(labels: Sequence[int]) -> list[list[int]]:
    """Cluster labels as index groups, in order. Singletons are kept: a
    non-degenerate state still has a phase, and that phase is also gauge."""
    groups: list[list[int]] = []
    for index, label in enumerate(labels):
        if groups and label == labels[index - 1]:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


# --------------------------------------------------------------------------- #
# The metrics themselves
# --------------------------------------------------------------------------- #


def subspace_overlap(C_row: Any, S: Any, C_col: Any) -> np.ndarray:
    """``M = C_row† S C_col``. ``S`` may be dense or sparse; only ``S @ C`` is used."""
    left = np.asarray(C_row)
    right = np.asarray(C_col)
    if left.ndim != 2 or right.ndim != 2:
        raise SubspaceGateError(f"coefficients must be 2-D, got {left.shape} and {right.shape}")
    if left.shape[0] != right.shape[0]:
        raise SubspaceGateError(
            f"windows live in different bases: {left.shape[0]} vs {right.shape[0]} orbitals"
        )
    return np.asarray(left.conj().T @ (S @ right))


def subspace_metrics(M: Any) -> dict[str, Any]:
    """Principal angles and projector distance of two S-orthonormal windows.

    ``M`` is :func:`subspace_overlap`'s output. The identification
    ``sigma_i = cos(theta_i)`` holds *because* both bases are S-orthonormal; on a
    window C15 rejected it means nothing, which is why the gate validates
    orthonormality first.
    """
    matrix = np.asarray(M)
    if matrix.ndim != 2:
        raise SubspaceGateError(f"the cross-Gram matrix must be 2-D, got {matrix.shape}")
    dim_row, dim_col = matrix.shape
    singular = np.linalg.svd(matrix, compute_uv=False) if matrix.size else np.zeros(0)
    angles = np.arccos(np.clip(singular, 0.0, 1.0))
    overlap_trace = float(np.sum(singular**2))
    return {
        "dimension_row": int(dim_row),
        "dimension_column": int(dim_col),
        "singular_values": singular.tolist(),
        "principal_angles_rad": angles.tolist(),
        "maximum_principal_angle_rad": float(angles.max(initial=0.0)),
        "projector_trace_overlap": overlap_trace,
        "projector_frobenius_distance": float(
            np.sqrt(max(dim_row + dim_col - 2.0 * overlap_trace, 0.0))
        ),
    }


def subspace_metric_tolerances(
    identity_error_row: float,
    identity_error_column: float,
    *,
    dimension: int,
    orbital_count: int,
) -> dict[str, Any]:
    """What :func:`subspace_metrics` may be trusted to, by :data:`METRIC_TOLERANCE_POLICY`."""
    floor = orbital_count * float(np.finfo(np.float64).eps)
    delta = max(float(identity_error_row), floor) + max(float(identity_error_column), floor)
    return {
        "identity_error": delta,
        "singular_value": delta,
        "principal_angle_rad": float(np.sqrt(2.0 * delta)),
        "projector_frobenius_distance": float(np.sqrt(4.0 * max(int(dimension), 1) * delta)),
        "policy": METRIC_TOLERANCE_POLICY,
    }


def block_metrics(
    values: Any, row_labels: Sequence[int], column_labels: Sequence[int]
) -> list[dict[str, Any]]:
    """Cluster-resolved, gauge-invariant metrics of a coupling matrix.

    Under a block-diagonal gauge the ``(a, b)`` block goes to ``U_a† g_ab U_b``,
    so its singular values and Frobenius norm survive while its entries do not.
    Those are exactly what :func:`epc_formalism.gauge_invariant_block_metrics`
    publishes; this only cuts the matrix along the clusters.
    """
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape != (len(row_labels), len(column_labels)):
        raise SubspaceGateError(
            f"block shape {matrix.shape} does not match "
            f"{len(row_labels)}x{len(column_labels)} cluster labels"
        )
    rows = cluster_groups(row_labels)
    columns = cluster_groups(column_labels)
    return [
        {
            "row_cluster": int(row_labels[row[0]]),
            "column_cluster": int(column_labels[column[0]]),
            "row_states": list(row),
            "column_states": list(column),
            **gauge_invariant_block_metrics(matrix[np.ix_(row, column)]),
        }
        for row in rows
        for column in columns
    ]


def random_gauge_rotation(
    groups: Sequence[Sequence[int]], dimension: int, generator: np.random.Generator
) -> np.ndarray:
    """Block-diagonal unitary: an independent Haar-distributed rotation per group.

    A 1x1 group gives a random phase, which is the gauge freedom of a
    non-degenerate state.
    """
    rotation = np.eye(int(dimension), dtype=np.complex128)
    for group in groups:
        size = len(group)
        block = generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
        unitary, upper = np.linalg.qr(block)
        unitary = unitary * (np.diag(upper) / np.abs(np.diag(upper)))[None, :]
        rotation[np.ix_(list(group), list(group))] = unitary
    return rotation


def cross_cluster_mixing(
    groups: Sequence[Sequence[int]], dimension: int, angle: float = np.pi / 4
) -> np.ndarray:
    """A unitary that is *not* a gauge: it rotates one state of the first cluster
    into one state of the second. The teeth of the invariance check — the same
    metrics must move by far more than their tolerance."""
    if len(groups) < 2:
        raise SubspaceGateError("cross-cluster mixing needs at least two clusters")
    rotation = np.eye(int(dimension), dtype=np.complex128)
    first, second = groups[0][-1], groups[1][0]
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation[first, first] = rotation[second, second] = cosine
    rotation[second, first], rotation[first, second] = sine, -sine
    return rotation


# --------------------------------------------------------------------------- #
# The gate, on the persisted eigenspaces
# --------------------------------------------------------------------------- #


def window_metric(payload: dict) -> np.ndarray:
    """``C†SC`` of one persisted window, from the deviation the solver stored.

    This is why the gate needs no ``S``: inside the window, ``I + D`` *is* the
    metric, so a gauge rotation of the window can be certified from C15's own
    artifacts.
    """
    deviation = np.asarray(payload["overlap_minus_identity"])
    return np.eye(deviation.shape[0], dtype=np.complex128) + deviation


def certify_window(payload: dict, *, seed: int = 0, separation_factor: float = 1.0) -> dict:
    """GO-6 on one persisted k point: cluster, rotate, measure, and try to break it."""
    energies = np.asarray(payload["energies_eV"], dtype=np.float64)
    residuals = np.asarray(payload["generalized_relative_residual"], dtype=np.float64)
    condition = float(payload["window_metric_condition_number"])
    orbital_count = int(payload["norbits"])
    metric = window_metric(payload)
    states = metric.shape[0]

    resolutions = eigenvalue_resolutions_ev(energies, residuals, condition, orbital_count)
    labels = near_degenerate_clusters(energies, resolutions, separation_factor=separation_factor)
    groups = cluster_groups(labels)
    identity_error = float(np.abs(metric - np.eye(states)).max(initial=0.0))
    tolerances = subspace_metric_tolerances(
        identity_error,
        identity_error,
        dimension=max((len(group) for group in groups), default=1),
        orbital_count=orbital_count,
    )

    # C -> C U with U block diagonal in the clusters: every cluster's span is
    # fixed, so every principal angle between a cluster and its rotation is zero.
    gauge = random_gauge_rotation(groups, states, np.random.default_rng(seed))
    rotated = metric @ gauge
    per_cluster = []
    for group in groups:
        index = np.ix_(list(group), list(group))
        measured = subspace_metrics(rotated[index])
        per_cluster.append({
            "cluster": int(labels[group[0]]),
            "states": list(group),
            "energies_eV": energies[list(group)].tolist(),
            "resolutions_eV": resolutions[list(group)].tolist(),
            "energy_spread_eV": float(energies[list(group)].max() - energies[list(group)].min()),
            **measured,
            "gauge_invariant": (
                measured["maximum_principal_angle_rad"] <= tolerances["principal_angle_rad"]
                and measured["projector_frobenius_distance"]
                <= tolerances["projector_frobenius_distance"]
            ),
        })

    # Teeth: mixing across a *resolved* gap is not a gauge, and must be seen.
    if len(groups) > 1:
        mixed = metric @ cross_cluster_mixing(groups, states)
        group = list(groups[0])
        measured = subspace_metrics(mixed[np.ix_(group, group)])
        detected = measured["maximum_principal_angle_rad"] > 10.0 * tolerances["principal_angle_rad"]
        discrimination = {
            "applicable": True,
            "detected": bool(detected),
            "maximum_principal_angle_rad": measured["maximum_principal_angle_rad"],
            "projector_frobenius_distance": measured["projector_frobenius_distance"],
        }
    else:
        discrimination = {
            "applicable": False,
            "detected": False,
            "detail": "the whole window is one unresolved cluster; nothing to mix across",
        }

    return {
        "k_index": int(payload["k_index"]),
        "k_fractional": [float(value) for value in payload["k_fractional"]],
        "state_count": states,
        "orbital_count": orbital_count,
        "window_metric_condition_number": condition,
        "maximum_generalized_relative_residual": float(residuals.max(initial=0.0)),
        "identity_maximum_absolute_error": identity_error,
        "cluster_labels": labels,
        "cluster_count": len(groups),
        "resolutions_eV": resolutions.tolist(),
        "tolerances": tolerances,
        "clusters": per_cluster,
        "cross_cluster_discrimination": discrimination,
        "passed": all(entry["gauge_invariant"] for entry in per_cluster)
        and (discrimination["detected"] or not discrimination["applicable"]),
    }


def certify(
    output_dir: Path,
    kpoint_count: int,
    *,
    seed: int = 0,
    separation_factor: float = 1.0,
    degeneracy_ev: float = DEFAULT_EIGENSPACE_DEGENERACY_EV,
) -> dict[str, Any]:
    """Run GO-6 over a C15 eigenspace directory."""
    _, diagnostics = persisted_eigenspace_data(
        output_dir, kpoint_count, degeneracy_ev=degeneracy_ev
    )
    windows = [
        certify_window(
            load_persisted_eigenspace(output_dir, k_index),
            seed=seed + k_index,
            separation_factor=separation_factor,
        )
        for k_index in range(kpoint_count)
    ]
    degenerate = [row for row in windows if row["cross_cluster_discrimination"]["applicable"]]
    return {
        "schema": SCHEMA,
        "gate": "GO-6",
        "gate_id": GATE_ID,
        "ticket": "C16 / E-F_001-S19",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eigenspace_dir": str(output_dir),
        "eigenspace_status": diagnostics["status"],
        "eigenspace_identity_tolerance_policy": diagnostics["identity_tolerance_policy"],
        "energy_resolution_policy": ENERGY_RESOLUTION_POLICY,
        "metric_tolerance_policy": METRIC_TOLERANCE_POLICY,
        "separation_factor": float(separation_factor),
        "gauge_seed": int(seed),
        "windows": windows,
        "summary": {
            "kpoint_count": len(windows),
            "windows_passed": sum(1 for row in windows if row["passed"]),
            "windows_with_a_resolved_gap": len(degenerate),
            "maximum_principal_angle_rad": max(
                (
                    entry["maximum_principal_angle_rad"]
                    for row in windows
                    for entry in row["clusters"]
                ),
                default=0.0,
            ),
            "failed_k_indices": [row["k_index"] for row in windows if not row["passed"]],
        },
        # A directory whose every window is one unresolved cluster exercises no
        # discrimination, so it cannot certify the machinery either.
        "verdict": "PASS" if windows and all(row["passed"] for row in windows) and degenerate
        else "NO_GO",
    }


def require_go6(report: Any) -> dict[str, Any]:
    """Gate for downstream K/TBG work: load and enforce a GO-6 PASS.

    ``report`` is the certification dict or the path it was written to.
    """
    if isinstance(report, (str, Path)):
        path = Path(report)
        if not path.is_file():
            raise SubspaceGateError(f"GO-6 has not been certified: {path} does not exist")
        report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("gate_id") != GATE_ID:
        raise SubspaceGateError(f"not a {GATE_ID} report: gate_id={report.get('gate_id')!r}")
    if report.get("verdict") != "PASS":
        raise SubspaceGateError(
            f"GO-6 is {report.get('verdict')}; subspace metrics must pass before "
            f"graphene K or any TBG validation (failed k: "
            f"{report.get('summary', {}).get('failed_k_indices')})"
        )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eigenspace-dir", type=Path, required=True, help="C15 solver output")
    parser.add_argument("--kpoint-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0, help="gauge rotation seed")
    parser.add_argument(
        "--separation-factor",
        type=float,
        default=1.0,
        help="a gap counts as resolved above this many resolutions",
    )
    parser.add_argument("--degeneracy-ev", type=float, default=DEFAULT_EIGENSPACE_DEGENERACY_EV)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = certify(
        args.eigenspace_dir,
        args.kpoint_count,
        seed=args.seed,
        separation_factor=args.separation_factor,
        degeneracy_ev=args.degeneracy_ev,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / REPORT_NAME
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"GO-6 subspace metrics: {report['verdict']}  ->  {output}")
    for row in report["windows"]:
        print(
            f"  k={row['k_index']:<3} clusters={row['cluster_count']:<3} "
            f"max angle={max((entry['maximum_principal_angle_rad'] for entry in row['clusters']), default=0.0):.3e} "
            f"tol={row['tolerances']['principal_angle_rad']:.3e} "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
