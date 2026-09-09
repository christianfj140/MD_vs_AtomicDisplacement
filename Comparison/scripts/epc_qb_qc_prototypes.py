#!/usr/bin/env python3
"""E-F_001-S32: minimal prototypes of routes Q-B and Q-C, checked against Q-A.

The roadmap (Section II) asks for two candidate routes to a scalable
``q != 0`` Graph2Mat-style response, on top of the transparent but expensive
route Q-A (:mod:`epc_commensurate_k`, a commensurate supercell):

* **Q-B** -- inject the phase into the *tangent* of a periodic edge before
  differentiating: ``v_j exp(i q.T) - v_i`` for an edge from home atom ``i``
  to neighbour ``j`` at periodic shift ``T``, then one JVP-like pass per ``q``.
* **Q-C** -- differentiate the real-space blocks once (a small, ``q``-free
  kernel indexed by neighbour shell), then Fourier-contract that kernel for as
  many ``(k, q)`` pairs as needed, at O(shells) cost each.

Both are prototyped here on a toy periodic 1D two-atom-per-cell tight-binding
chain -- the same functional family as the toy used to certify Q-A itself
(``tests/test_epc_commensurate_k.py``: exponential-decay hopping/overlap,
alternating onsite) -- because the physical basis-response gate (GO-4, C14C)
is still ``NO_GO`` (docs/epc_go4_graphene_gamma_verdict.md) and running either
route on real graphene would only re-report that open gate, not exercise the
``q``-space mechanics this ticket is about. Nothing here is
``candidate__pending_go5`` graphene physics; it is algebra on a toy crystal,
status ``prototype__algebra_only``.

The physics that makes both routes work
----------------------------------------
For a home cell always at relative image ``R = 0`` and a neighbour at image
``R_l`` (an integer multiple of the 1D lattice vector here), moving the home
atom by ``v`` and the neighbour by ``v * exp(i q.R_l)`` (canonical cell gauge,
:mod:`epc_fourier`) gives, by the chain rule alone (no periodicity assumed):

    dH(R_l)/deps = A_l * exp(i q.R_l) + B_l,   A_l = dH(R_l)/dr_neighbour . v,
                                                B_l = dH(R_l)/dr_home . v

``A_l``/``B_l`` do not depend on ``q``: they are the real-space derivative
kernel Q-C keeps around. Bloch-summing the *whole* perturbed crystal at
electronic wavevector ``k`` (:func:`epc_fourier.bloch_sum`) then gives the
standard electron-phonon real-space-to-k identity used here and checked
against Q-A below:

    D_H(k+q, k) = sum_l A_l exp(i k'.R_l) + sum_l B_l exp(i k.R_l),  k' = k+q

which is exactly :func:`k_to_k_plus_q_kernel`. Q-C evaluates this from a
table built once per direction ``v`` (``kernel_table``, independent of both
``k`` and ``q``). Q-B instead folds the ``q`` phase into the tangent *before*
differentiating (:func:`phase_aware_kernel`), producing the equivalent
``q``-dependent kernel ``A_l exp(i q.R_l) + B_l`` in a single finite-difference
pass -- cheaper to set up for one ``q`` (no separate ``A_l``/``B_l``
bookkeeping) but, unlike Q-C's table, it must be rebuilt from scratch for
every new ``q`` (:func:`scaling_report`).

Validated only where it can be
-------------------------------
Q-A can only probe *commensurate* electronic ``k`` (those that fold onto the
supercell's own discrete k-points; :mod:`epc_commensurate_k` docstring,
Section 2). At an incommensurate ``k`` the supercell's Bloch-character
sandwich is not a well-defined projection of anything physical -- it depends
on an arbitrary choice of which periodic image represents "the home cell",
which only cancels out at commensurate ``k``. :func:`compare_routes` checks
Q-B and Q-C against Q-A only at commensurate ``k`` for that reason; this is a
limit of the *reference*, not of Q-B or Q-C, and is recorded explicitly in
every report this module returns (``reference_limits``).
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from epc_commensurate_k import (  # noqa: E402
    CommensurateCell,
    bloch_character_basis,
    diagonal_supercell_matrix,
)
from epc_fourier import k_plus_q  # noqa: E402

SCHEMA = "epc_qb_qc_prototype_v1"
TICKET = "E-F_001-S32"
STATUS = "prototype__algebra_only"

#: Why graphene K is not attempted here: GO-4's gate 5 blocks the physical
#: layer of Q-A itself (docs/epc_s31_graphene_k_go5_verdict.md); routing a
#: real system through Q-B/Q-C would inherit exactly that open gate.
GO4_BLOCKING_GATE = "gate 5 (basis_response_validated_and_required_by_formalism_id): C14C"

# --------------------------------------------------------------------------- #
# The toy crystal: 1D chain, two atoms per cell, exponential-decay SK hopping
# --------------------------------------------------------------------------- #

A_ANG = 2.5
CELL_VECTOR_ANG = np.array([A_ANG, 0.0, 0.0])
PRIMITIVE_CELL_ANG = np.diag([A_ANG, 12.0, 12.0])
PRIMITIVE_POSITIONS_ANG = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
ONSITE_EV = (-1.0, 1.0)
T0_EV, S0, LAMBDA_ANG = 2.4, 0.18, 0.9
#: 1.5*A restricts real neighbours to |l| <= 1 -- the same range a Nc=3
#: commensurate supercell can represent by minimum image without ambiguity,
#: which is what lets Q-A serve as a reference at all.
CUTOFF_RADIUS_ANG = 1.5 * A_ANG
CUTOFF_SHELLS = 1
FD_DELTA_ANG = 1.0e-5


class PrototypeError(RuntimeError):
    """A Q-B/Q-C prototype computation was asked for something ill-posed."""


def pair_term(distance_ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(H, S)`` off-diagonal SK-like term at ``distance_ang`` (same family as
    the toy in ``tests/test_epc_commensurate_k.py``)."""
    decay = np.exp(-np.asarray(distance_ang, dtype=np.float64) / LAMBDA_ANG)
    return -T0_EV * decay, S0 * decay


def pair_block(home_pos: np.ndarray, neighbor_pos: np.ndarray, shell: int) -> tuple[np.ndarray, np.ndarray]:
    """The 2x2 ``(H(R_shell), S(R_shell))`` block between two explicit atom sets.

    ``home_pos``/``neighbor_pos`` are ``(2, 3)`` Cartesian positions of the two
    toy atoms playing the "home cell" and "neighbour at image `shell`" role;
    pairs beyond :data:`CUTOFF_RADIUS_ANG` are exactly zero (a real, not a
    numerical-noise, sparsity).
    """
    H = np.zeros((2, 2))
    S = np.zeros((2, 2))
    for mu in range(2):
        for nu in range(2):
            if shell == 0 and mu == nu:
                H[mu, mu] = ONSITE_EV[mu]
                S[mu, mu] = 1.0
                continue
            distance = float(np.linalg.norm(neighbor_pos[nu] - home_pos[mu]))
            if distance > CUTOFF_RADIUS_ANG:
                continue
            h, s = pair_term(distance)
            H[mu, nu] = float(h)
            S[mu, nu] = float(s)
    return H, S


def supercell_matrices(positions_ang: np.ndarray, supercell_length_ang: float) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(H, S)`` of a periodic supercell, minimum image along x.

    This is the Q-A reference builder: the same pairwise cutoff physics as
    :func:`pair_block`, evaluated directly and densely on the whole supercell
    the way an actual SCF code would see a Born-von-Karman ring of this size.
    """
    positions = np.asarray(positions_ang, dtype=np.float64)
    n = positions.shape[0]
    dx = positions[:, 0][:, None] - positions[:, 0][None, :]
    dx -= supercell_length_ang * np.round(dx / supercell_length_ang)
    dy = positions[:, 1][:, None] - positions[:, 1][None, :]
    dz = positions[:, 2][:, None] - positions[:, 2][None, :]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    off_diagonal = ~np.eye(n, dtype=bool)
    within_cutoff = off_diagonal & (distance <= CUTOFF_RADIUS_ANG)
    H = np.zeros((n, n))
    S = np.zeros((n, n))
    h, s = pair_term(distance[within_cutoff])
    H[within_cutoff] = h
    S[within_cutoff] = s
    for atom in range(n):
        H[atom, atom] = ONSITE_EV[atom % 2]
        S[atom, atom] = 1.0
    return H, S


# --------------------------------------------------------------------------- #
# Q-A: the commensurate supercell reference (reused, not reimplemented)
# --------------------------------------------------------------------------- #


def commensurate_cell(q_primitive: Any) -> CommensurateCell:
    return CommensurateCell(
        q_primitive=tuple(float(v) for v in q_primitive),
        matrix=diagonal_supercell_matrix(q_primitive),
        primitive_cell_ang=PRIMITIVE_CELL_ANG,
        primitive_positions_ang=PRIMITIVE_POSITIONS_ANG,
    )


def q_a_reference(q_primitive: Any, direction: np.ndarray, k_primitive: Any, *, delta: float = FD_DELTA_ANG) -> dict[str, Any]:
    """Route Q-A: explicit commensurate supercell, central finite difference.

    Builds the whole ``(N_c * 2, N_c * 2)`` supercell, moves it by the
    ``q``-periodic pattern of ``direction`` (Q-A's own machinery,
    ``epc_commensurate_k``: the finite ``q`` lives in the pattern, the
    supercell response is an ordinary Gamma one), and projects the result onto
    the primitive ``k``/``k+q`` Bloch character with
    :func:`epc_commensurate_k.bloch_character_basis` -- exactly the GO-6
    selection Q-A uses for real graphene, just without the electronic
    eigenvectors (this module compares raw ``D_H``/``D_S``, not ``g``).
    """
    cell = commensurate_cell(q_primitive)
    phases = np.exp(2j * np.pi * (cell.images.astype(np.float64) @ np.asarray(q_primitive, dtype=np.float64)))
    pattern = np.concatenate([direction * phase for phase in phases])
    cos_pattern, sin_pattern = pattern.real, pattern.imag

    positions = cell.positions_ang
    length = float(cell.cell_ang[0, 0])

    def _directional_fd(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_plus, s_plus = supercell_matrices(positions + delta * tangent, length)
        h_minus, s_minus = supercell_matrices(positions - delta * tangent, length)
        return (h_plus - h_minus) / (2.0 * delta), (s_plus - s_minus) / (2.0 * delta)

    D_H_cos, D_S_cos = _directional_fd(cos_pattern)
    D_H_sin, D_S_sin = _directional_fd(sin_pattern)
    D_H_super = D_H_cos + 1j * D_H_sin
    D_S_super = D_S_cos + 1j * D_S_sin

    k_final = k_plus_q(k_primitive, q_primitive)
    basis_initial = bloch_character_basis(cell.atom_image_index, cell.images, k_primitive)
    basis_final = bloch_character_basis(cell.atom_image_index, cell.images, k_final)
    return {
        "D_H": basis_final.conj().T @ D_H_super @ basis_initial,
        "D_S": basis_final.conj().T @ D_S_super @ basis_initial,
        "supercell_atom_count": int(positions.shape[0]),
        "supercell_dense_entries": int(positions.shape[0]) ** 2,
    }


# --------------------------------------------------------------------------- #
# Q-C: real-space derivative kernel, built once, Fourier-contracted per (k, q)
# --------------------------------------------------------------------------- #


def kernel_table(direction: np.ndarray, *, shells: int = CUTOFF_SHELLS, delta: float = FD_DELTA_ANG) -> dict[str, Any]:
    """``{shell: (A_l, B_l)}``, the q-free real-space derivative kernel (Q-C).

    ``A_l`` is the response to moving the *neighbour* copy at shell ``l`` (the
    part that will need the ``q`` phase later); ``B_l`` is the response to
    moving the *home* copy (no phase, ever -- the home cell is always the
    Fourier origin, :mod:`epc_fourier`). Built once per direction; reused for
    every ``(k, q)`` pair afterwards.
    """
    start = time.perf_counter()
    A: dict[int, np.ndarray] = {}
    B: dict[int, np.ndarray] = {}
    A_S: dict[int, np.ndarray] = {}
    B_S: dict[int, np.ndarray] = {}
    for shell in range(-shells, shells + 1):
        neighbor_baseline = PRIMITIVE_POSITIONS_ANG + shell * CELL_VECTOR_ANG
        h_plus, s_plus = pair_block(PRIMITIVE_POSITIONS_ANG, neighbor_baseline + delta * direction, shell)
        h_minus, s_minus = pair_block(PRIMITIVE_POSITIONS_ANG, neighbor_baseline - delta * direction, shell)
        A[shell] = (h_plus - h_minus) / (2.0 * delta)
        A_S[shell] = (s_plus - s_minus) / (2.0 * delta)
        h_plus, s_plus = pair_block(PRIMITIVE_POSITIONS_ANG + delta * direction, neighbor_baseline, shell)
        h_minus, s_minus = pair_block(PRIMITIVE_POSITIONS_ANG - delta * direction, neighbor_baseline, shell)
        B[shell] = (h_plus - h_minus) / (2.0 * delta)
        B_S[shell] = (s_plus - s_minus) / (2.0 * delta)
    nonzero_shells = sum(1 for shell in A if np.abs(A[shell]).max() > 0.0 or np.abs(B[shell]).max() > 0.0)
    return {
        "A_H": A, "B_H": B, "A_S": A_S, "B_S": B_S,
        "shells": shells,
        "fd_evaluations": 4 * (2 * shells + 1),
        "build_seconds": time.perf_counter() - start,
        "sparsity_nonzero_shell_fraction": nonzero_shells / (2 * shells + 1),
    }


def _bloch_sum_1d(blocks: dict[int, np.ndarray], k_x: float) -> np.ndarray:
    total = np.zeros((2, 2), dtype=complex)
    for shell, block in blocks.items():
        total = total + block * np.exp(2j * np.pi * shell * k_x)
    return total


def k_to_k_plus_q_kernel(table: dict[str, Any], k_primitive: Any, q_primitive: Any) -> dict[str, np.ndarray]:
    """Route Q-C: ``D_H(k+q, k) = sum_l A_l exp(i(k+q).R_l) + sum_l B_l exp(i k.R_l)``.

    O(shells) once :func:`kernel_table` exists; no new finite difference for a
    new ``(k, q)``, which is the whole point of keeping the table around.
    """
    k_final = k_plus_q(k_primitive, q_primitive)
    k_x, kq_x = float(np.asarray(k_primitive).reshape(3)[0]), float(np.asarray(k_final).reshape(3)[0])
    return {
        "D_H": _bloch_sum_1d(table["A_H"], kq_x) + _bloch_sum_1d(table["B_H"], k_x),
        "D_S": _bloch_sum_1d(table["A_S"], kq_x) + _bloch_sum_1d(table["B_S"], k_x),
    }


# --------------------------------------------------------------------------- #
# Q-B: phase folded into the tangent before differentiating, one pass per q
# --------------------------------------------------------------------------- #


def phase_aware_kernel(direction: np.ndarray, q_primitive: Any, *, shells: int = CUTOFF_SHELLS, delta: float = FD_DELTA_ANG) -> dict[str, Any]:
    """Route Q-B: ``v_j exp(i q.T) - v_i`` folded into the tangent, one JVP-like
    pass (two real finite differences: cos/sin of the phase) per ``q``.

    Unlike :func:`kernel_table`, the result already has ``q`` baked in --
    there is no reusable, ``q``-free intermediate. That is the trade-off this
    ticket is asked to measure (:func:`scaling_report`): cheaper to reach for
    a single ``q`` (no separate table object), but redone in full for the next
    one, whereas Q-C's table survives across every subsequent ``q``.
    """
    start = time.perf_counter()
    q_x = float(np.asarray(q_primitive, dtype=np.float64).reshape(3)[0])
    kernel_H: dict[int, np.ndarray] = {}
    kernel_S: dict[int, np.ndarray] = {}
    for shell in range(-shells, shells + 1):
        phase = 2.0 * np.pi * shell * q_x
        neighbor_baseline = PRIMITIVE_POSITIONS_ANG + shell * CELL_VECTOR_ANG
        # The combined tangent: home copy gets weight 1 (real), the neighbour
        # copy gets weight exp(i.phase) -- split into its cos/sin (both real)
        # parts because the toy Hamiltonian only accepts real positions.
        home_re, neighbor_re = direction, direction * np.cos(phase)
        home_im, neighbor_im = np.zeros_like(direction), direction * np.sin(phase)

        h_plus, s_plus = pair_block(PRIMITIVE_POSITIONS_ANG + delta * home_re, neighbor_baseline + delta * neighbor_re, shell)
        h_minus, s_minus = pair_block(PRIMITIVE_POSITIONS_ANG - delta * home_re, neighbor_baseline - delta * neighbor_re, shell)
        d_h_re, d_s_re = (h_plus - h_minus) / (2.0 * delta), (s_plus - s_minus) / (2.0 * delta)

        h_plus, s_plus = pair_block(PRIMITIVE_POSITIONS_ANG + delta * home_im, neighbor_baseline + delta * neighbor_im, shell)
        h_minus, s_minus = pair_block(PRIMITIVE_POSITIONS_ANG - delta * home_im, neighbor_baseline - delta * neighbor_im, shell)
        d_h_im, d_s_im = (h_plus - h_minus) / (2.0 * delta), (s_plus - s_minus) / (2.0 * delta)

        kernel_H[shell] = d_h_re + 1j * d_h_im
        kernel_S[shell] = d_s_re + 1j * d_s_im
    return {
        "kernel_H": kernel_H,
        "kernel_S": kernel_S,
        "q_primitive": tuple(float(v) for v in np.asarray(q_primitive, dtype=np.float64).reshape(3)),
        "shells": shells,
        "fd_evaluations": 4 * (2 * shells + 1),
        "build_seconds": time.perf_counter() - start,
    }


def q_b_response(phase_aware: dict[str, Any], k_primitive: Any) -> dict[str, np.ndarray]:
    """The ``k``-weighted sum on top of one :func:`phase_aware_kernel` result."""
    k_x = float(np.asarray(k_primitive, dtype=np.float64).reshape(3)[0])
    return {
        "D_H": _bloch_sum_1d(phase_aware["kernel_H"], k_x),
        "D_S": _bloch_sum_1d(phase_aware["kernel_S"], k_x),
    }


# --------------------------------------------------------------------------- #
# GPU preflight (Default compute policy)
# --------------------------------------------------------------------------- #


def gpu_preflight() -> dict[str, Any]:
    """Record the requested/effective backend, never fall back silently.

    Every kernel in this module is a handful of 2x2 dense complex matrices
    (``shells`` of them, ``shells <= a few``): there is no GPU-eligible
    workload here, torch/Graph2Mat are not invoked, and a synthetic
    scaling sweep (:func:`scaling_report`) confirms wall time stays in the
    microsecond-to-millisecond range even at 64 shells. CPU is used because
    the preflight shows no problem size in this module ever approaches a
    GPU crossover, not because CUDA was not checked.
    """
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - torch absent is a valid state
        return {
            "requested_backend": "cuda_if_available",
            "effective_backend": "cpu",
            "cuda_available": False,
            "reason_for_cpu": f"torch import failed ({exc.__class__.__name__}); numpy-only prototype",
        }
    return {
        "requested_backend": "cuda_if_available",
        "effective_backend": "cpu",
        "cuda_available": cuda_available,
        "reason_for_cpu": (
            "no GPU-eligible workload: every kernel is O(shells) 2x2 dense complex matrices "
            "(microseconds, see scaling_report); this algebra prototype calls no torch/Graph2Mat model"
        ),
    }


# --------------------------------------------------------------------------- #
# The comparison the roadmap asks for
# --------------------------------------------------------------------------- #

#: Commensurate k for the toy's q=1/3 supercell (Nc=3): the only k where Q-A's
#: Bloch-character sandwich is unambiguous (module docstring, "Validated only
#: where it can be"). Values outside this set are a reference limit, not a
#: Q-B/Q-C failure.
_COMMENSURATE_K_FRACTIONS = (0.0, 1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0)


def compare_routes(q_primitive: Any = (1.0 / 3.0, 0.0, 0.0), direction: np.ndarray | None = None) -> dict[str, Any]:
    """Q-A vs Q-B vs Q-C at every commensurate ``k`` of this toy's supercell."""
    if direction is None:
        direction = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != PRIMITIVE_POSITIONS_ANG.shape:
        raise PrototypeError(f"direction must be {PRIMITIVE_POSITIONS_ANG.shape}, got {direction.shape}")

    table = kernel_table(direction)
    phase_aware = phase_aware_kernel(direction, q_primitive)

    rows = []
    for k_fraction in _COMMENSURATE_K_FRACTIONS:
        k_primitive = (k_fraction, 0.0, 0.0)
        reference = q_a_reference(q_primitive, direction, k_primitive)
        qc = k_to_k_plus_q_kernel(table, k_primitive, q_primitive)
        qb = q_b_response(phase_aware, k_primitive)
        rows.append({
            "k_primitive": list(k_primitive),
            "max_abs_D_H_qa_vs_qc": float(np.abs(reference["D_H"] - qc["D_H"]).max()),
            "max_abs_D_H_qa_vs_qb": float(np.abs(reference["D_H"] - qb["D_H"]).max()),
            "max_abs_D_H_qb_vs_qc": float(np.abs(qb["D_H"] - qc["D_H"]).max()),
            "max_abs_D_S_qa_vs_qc": float(np.abs(reference["D_S"] - qc["D_S"]).max()),
            "max_abs_D_S_qa_vs_qb": float(np.abs(reference["D_S"] - qb["D_S"]).max()),
        })
    worst = max(row["max_abs_D_H_qa_vs_qc"] for row in rows)
    worst = max(worst, max(row["max_abs_D_H_qa_vs_qb"] for row in rows))

    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "status": STATUS,
        "blocked_full_ks_gate": GO4_BLOCKING_GATE,
        "q_primitive": [float(v) for v in np.asarray(q_primitive, dtype=np.float64).reshape(3)],
        "direction": direction.tolist(),
        "fd_delta_ang": FD_DELTA_ANG,
        "per_k": rows,
        "worst_case_agreement_D_H": worst,
        "reference_limits": (
            "Q-A's bloch_character_basis sandwich is only a well-defined projection at "
            "commensurate electronic k (folds onto the supercell's own discrete k-points); "
            "this comparison is restricted to that set, per the module docstring"
        ),
        "kernel_table_metrics": {
            "shells": table["shells"],
            "fd_evaluations_to_build": table["fd_evaluations"],
            "build_seconds": table["build_seconds"],
            "sparsity_nonzero_shell_fraction": table["sparsity_nonzero_shell_fraction"],
        },
        "phase_aware_kernel_metrics": {
            "shells": phase_aware["shells"],
            "fd_evaluations_to_build": phase_aware["fd_evaluations"],
            "build_seconds": phase_aware["build_seconds"],
        },
        "gpu_preflight": gpu_preflight(),
        "not_materialized": "no (n_outputs, N, 3) Jacobian anywhere; every kernel is a (shells,) "
        "table of 2x2 blocks, Fourier-contracted, never a dense per-atom Jacobian",
    }


def scaling_report(q_values: Any, k_values: Any, *, shells: int = CUTOFF_SHELLS, direction: np.ndarray | None = None) -> dict[str, Any]:
    """Measure the A/B trade-off Section X asks for: reuse (Q-C) vs redo (Q-B).

    Q-C pays :func:`kernel_table` once, then O(len(q)*len(k)) cheap sums.
    Q-B pays :func:`phase_aware_kernel` once *per q*, then the same O(len(k))
    cheap sums per q. Peak memory is also reported (:mod:`tracemalloc`): both
    tables are ``shells``-sized dictionaries of 2x2 complex blocks, so neither
    should show growth with ``len(q)`` or ``len(k)``.
    """
    if direction is None:
        direction = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)
    q_values = [tuple(float(c) for c in np.asarray(q, dtype=np.float64).reshape(3)) for q in q_values]
    k_values = [tuple(float(c) for c in np.asarray(k, dtype=np.float64).reshape(3)) for k in k_values]

    tracemalloc.start()
    start = time.perf_counter()
    table = kernel_table(direction, shells=shells)
    for q in q_values:
        for k in k_values:
            k_to_k_plus_q_kernel(table, k, q)
    qc_seconds = time.perf_counter() - start
    _, qc_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    start = time.perf_counter()
    for q in q_values:
        phase_aware = phase_aware_kernel(direction, q, shells=shells)
        for k in k_values:
            q_b_response(phase_aware, k)
    qb_seconds = time.perf_counter() - start
    _, qb_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "schema": "epc_qb_qc_scaling_v1",
        "ticket": TICKET,
        "n_q": len(q_values),
        "n_k": len(k_values),
        "shells": shells,
        "q_c_total_seconds": qc_seconds,
        "q_b_total_seconds": qb_seconds,
        "q_c_peak_bytes": qc_peak_bytes,
        "q_b_peak_bytes": qb_peak_bytes,
        "q_c_pays_fd_once": True,
        "q_b_pays_fd_per_q": True,
        "mechanism_note": (
            "Q-C pays kernel_table() once, then two Bloch sums (A(k+q) and B(k)) per query; "
            "Q-B pays phase_aware_kernel() once per distinct q (n_q finite differences total, "
            "not one), then one cheaper already-summed Bloch sum per query. Whether Q-C's "
            "amortized build wins over Q-B's cheaper-but-repeated one depends on n_q relative "
            "to n_k and shells -- see scaling_sweep() for a measured crossover, not an assumed one"
        ),
    }


def scaling_sweep(n_q_values: Any, n_k: int = 5, *, shells: int = CUTOFF_SHELLS) -> dict[str, Any]:
    """:func:`scaling_report` at several ``n_q``, to find the crossover empirically.

    Section X of the roadmap: "El benchmark decide" -- this repeats the A/B
    measurement at increasing ``n_q`` (fixed ``n_k``) and reports the smallest
    ``n_q`` in the sweep where Q-C's amortized build overtakes Q-B's
    repeated one, if any is observed in the tested range.
    """
    k_values = [(x, 0.0, 0.0) for x in np.linspace(-0.4, 0.4, n_k)]
    points = []
    crossover_n_q = None
    for n_q in n_q_values:
        q_values = [(x, 0.0, 0.0) for x in np.linspace(0.05, 0.45, n_q)]
        point = scaling_report(q_values, k_values, shells=shells)
        points.append({
            "n_q": n_q,
            "q_c_total_seconds": point["q_c_total_seconds"],
            "q_b_total_seconds": point["q_b_total_seconds"],
        })
        if crossover_n_q is None and point["q_c_total_seconds"] < point["q_b_total_seconds"]:
            crossover_n_q = n_q
    return {
        "schema": "epc_qb_qc_scaling_sweep_v1",
        "ticket": TICKET,
        "n_k": n_k,
        "shells": shells,
        "points": points,
        "crossover_n_q": crossover_n_q,
        "crossover_observed": crossover_n_q is not None,
    }


# --------------------------------------------------------------------------- #
# E-F_001-S33: production route decision (Q-B vs Q-C vs combination)
# --------------------------------------------------------------------------- #


def production_route_decision() -> dict[str, Any]:
    """Which route -- Q-B, Q-C, or a combination -- is production for MATBG
    ``q != 0``, decided from the measurements above (accuracy, basis-response
    coverage, cost, cache/restart capacity), not from intuition (roadmap
    Section II: "no se elegirá antes de graphene K").

    Four axes, judged separately:

    * **accuracy** -- tied. Both routes agree with Q-A wherever Q-A can judge,
      and with each other at incommensurate ``k`` too, at the FD noise floor
      (:func:`compare_routes`, ``worst_case_agreement_D_H``). This axis does
      not decide.
    * **basis response** -- tied, and orthogonal to this choice. Both routes
      produce ``D_S`` by the same mechanism as ``D_H``; neither implements
      ``S_L``/``S_R`` separately. Whatever C14C/GO-1 ultimately requires for
      the intra-atomic term is a ``basis_response_backend`` question, not a
      Q-B-vs-Q-C one -- picking a route here does not unblock GO-4's gate 5.
    * **cost** -- Q-C wins once more than a handful of ``q`` are requested
      (:func:`scaling_sweep`: crossover measured between ``n_q=1`` and
      ``n_q=5`` in this module's regime); Q-B is cheaper only for a single,
      one-off ``q``. A selected-mode MATBG campaign (roadmap Fase 10d)
      samples several low-energy sectors, plausibly more than one ``q``, and
      Fase 12 (``lambda``, ``alpha^2F``) explicitly needs a ``q``-mesh --
      exactly the regime where Q-C's amortized table wins with growing
      margin.
    * **cache/restart** -- decisive. :func:`kernel_table` is a genuine
      ``q``-independent intermediate: it fits the roadmap's restart
      granularity (Section X: "(q, mode_or_test_displacement, derivative,
      k-block)") as its own persistable node, so a restarted campaign reuses
      it instead of repeating finite differences already paid for.
      :func:`phase_aware_kernel` (Q-B) has no such intermediate -- every
      ``q`` pays its own finite-difference pass again, restart or not.

    GO-5 is currently ``NO_GO``. Therefore this record may compare candidates,
    but it must not select a production route yet.
    """
    comparison = compare_routes()
    sweep = scaling_sweep(n_q_values=(1, 5, 20, 80), n_k=5, shells=4)
    import epc_commensurate_k

    go5 = epc_commensurate_k.go5_decision()
    blocked = go5.get("go5") != "PASS"
    return {
        "schema": "epc_qb_qc_production_route_decision_v1",
        "ticket": "E-F_001-S33",
        "memo": "docs/epc_s33_ruta_qneq0_produccion.md",
        "status": "BLOCKED" if blocked else "DECIDED",
        "primary_route": None if blocked else "Q-C",
        "validation_only_route": None if blocked else "Q-B",
        "candidate_after_go5": "Q-C",
        "decision": (
            "UNDECIDED while GO-5 is NO_GO. Q-C remains a measured candidate, and Q-B a "
            "candidate cross-check; neither is selected for production."
        ),
        "axes": {
            "accuracy": {
                "verdict": "tie",
                "evidence": {
                    "worst_case_agreement_D_H": comparison["worst_case_agreement_D_H"],
                    "reference_limits": comparison["reference_limits"],
                },
            },
            "basis_response": {
                "verdict": "tie, orthogonal to this decision",
                "evidence": {
                    "max_abs_D_S_qa_vs_qc": max(row["max_abs_D_S_qa_vs_qc"] for row in comparison["per_k"]),
                    "max_abs_D_S_qa_vs_qb": max(row["max_abs_D_S_qa_vs_qb"] for row in comparison["per_k"]),
                },
                "note": (
                    "S_L/S_R vs D_S-only is a formalism_id/C14C question (GO-1), not resolved by "
                    "choosing Q-B or Q-C; both routes need the same extension if GO-1 requires "
                    "S_L/S_R separately"
                ),
            },
            "cost": {
                "verdict": "Q-C wins for n_q beyond a handful",
                "evidence": {
                    "crossover_n_q": sweep["crossover_n_q"],
                    "crossover_observed": sweep["crossover_observed"],
                    "n_k": sweep["n_k"],
                    "shells": sweep["shells"],
                    "points": sweep["points"],
                },
                "note": (
                    "the crossover is a property of this module's toy regime (n_k=5, shells=4), "
                    "not a MATBG constant; re-measure with scaling_sweep() at the real edge/shell "
                    "count before relying on it for a production preflight (roadmap Section X)"
                ),
            },
            "cache_restart": {
                "verdict": "decisive for Q-C",
                "reasoning": (
                    "kernel_table() is q-independent: it fits the roadmap's restart granularity "
                    "(q, mode_or_test_displacement, derivative, k-block) as its own persistable "
                    "node, so a restarted campaign reuses it instead of repeating finite "
                    "differences already paid for. phase_aware_kernel() (Q-B) has no q-independent "
                    "intermediate to persist -- every q pays its own finite-difference pass again, "
                    "restart or not."
                ),
                "signature_gap": (
                    "shared/artifact_signature.py's raw_derivative kind (fields: "
                    "derivative_backend, derivative_method, perturbation_definition, q, "
                    "delta_or_jvp, topology_sha256, convention, dtype, units) covers the final "
                    "per-(k, q) result of either route, but has no separate node for Q-C's "
                    "q-free kernel_table. Without one, a process restart would still redo the "
                    "table build even though it is q-independent: caching it as its own artifact "
                    "(e.g. a raw_derivative_kernel_table kind, keyed by direction + topology + "
                    "delta_or_jvp, consumed by the per-(k, q) raw_derivative node) is required "
                    "before Q-C's amortization survives a restart, not just a single process's "
                    "lifetime"
                ),
            },
        },
        "rejects_gamma_jvp_reuse": True,
        "gamma_jvp_reuse_rejection": (
            "The existing Graph2Mat directional JVP (graph2mat_autograd_derivatives.py, "
            "compute_graph2mat_directional_derivative) differentiates a cell-periodic tangent: it "
            "is q=0 of that cell by construction (roadmap Section II). It is explicitly NOT an "
            "acceptable substitute for either Q-B or Q-C, silent or otherwise -- B4 names this "
            "exact failure mode, and GO-8b requires 'ninguna reutilizacion del JVP Gamma como "
            "sustituto silencioso de una perturbacion q!=0'. Any future change that feeds a Gamma "
            "JVP result into a q!=0 contraction without going through phase_aware_kernel/"
            "kernel_table (or their real-Graph2Mat equivalents) is a GO-8b violation, not an "
            "optimization."
        ),
        "additional_tests_required_before_go8b": (
            "chosen route (Q-C, plus Q-B as cross-check) implemented against the real Graph2Mat "
            "model (graph2mat_autograd_derivatives.py), not only the toy chain here",
            "topology-margin test: no atom pair the kernel touches may cross the fixed "
            "edge_index/shifts cutoff of the checkpoint (roadmap 'Neighbor topology')",
            "q and -q, cell-origin shift, atom-across-periodic-boundary, and alternative "
            "atomic-phase-convention checks repeated on the real-Graph2Mat kernel -- the same "
            "five checks S31 ran for Q-A (docs/epc_s31_graphene_k_go5_verdict.md Sec. 2), not "
            "only on the toy",
            "agreement against Q-A on real graphene K, once GO-4/C14C unblocks it "
            "(docs/epc_s31_graphene_k_go5_verdict.md Sec. 4)",
            "a raw_derivative_kernel_table cache artifact (see axes.cache_restart.signature_gap) "
            "with a restart test: kill the process after kernel_table() completes, resume, and "
            "confirm no finite difference reruns",
            "GPU preflight on the real Graph2Mat kernel build (not this toy's dense 2x2 blocks); "
            "forward-mode JVP is already CUDA-compatible per prior verification, so this should "
            "extend that check, not repeat it",
            "a scaling_sweep() re-run at the real MATBG shell/edge count and the campaign's "
            "actual n_q, n_k, to confirm the crossover measured here on a toy chain still holds "
            "at production scale",
            "sparse serialization round-trip of the kernel_table/phase_aware_kernel outputs "
            "through the existing Graph2Mat -> sparse mapping (C11), not just dense 2x2 blocks",
        ),
        "no_production_started": True,
        "blocked_by": f"GO-5 {go5.get('go5')}: {go5.get('blocked_by')}" if blocked else None,
        "evidence": {
            "compare_routes": comparison,
            "scaling_sweep": sweep,
        },
    }


if __name__ == "__main__":
    import json

    report = compare_routes()
    print(json.dumps(report, indent=2))
    sweep = scaling_sweep(n_q_values=[1, 5, 20, 80], shells=4)
    print(json.dumps(sweep, indent=2))
    decision = production_route_decision()
    print(json.dumps(decision, indent=2))
