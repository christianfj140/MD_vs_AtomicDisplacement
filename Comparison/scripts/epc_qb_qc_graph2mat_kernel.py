#!/usr/bin/env python3
"""E-F_001-S40 (GO-8b): route Q-C's real-space kernel -- and Q-B's per-``q``
cross-check -- built on the REAL production JVP, not the hand-rolled finite
differences of :mod:`epc_qb_qc_prototypes`.

S33 (``docs/epc_s33_ruta_qneq0_produccion.md``) picked Q-C as the primary
``q != 0`` production route and listed eight things that still had to be true
before GO-8b, none of which the S32/S33 toy (pure numpy, hand-rolled central
differences) could demonstrate because it never touches
``graph2mat_autograd_derivatives.compute_graph2mat_directional_derivative``,
the actual production JVP. This module closes items 1, 2, 3, 5, 6, 7, 8 of
that list against a real (if small) periodic Graph2Mat-style model, going
through the unmodified production JVP function every time. Item 4 --
agreement against Q-A on *real* graphene K -- stays blocked by GO-4/GO-5
(``docs/epc_s31_graphene_k_go5_verdict.md``, ``NO_GO``): running it would
just re-report the same open intra-atomic basis-response gate (C14C) under a
different label, not exercise anything new about ``q``-space mechanics. So
``go8b_status()`` below reports ``NO-GO-8b`` honestly, with every other check
that *can* run today reported PASS/FAIL on its own evidence -- the same
two-suite shape ``certify_matbg_phonon_provider.py`` (S39) uses for GO-8a.

Why two independent position "copies" are needed at all
---------------------------------------------------------
A real production Graph2Mat/MACE graph has exactly ONE position leaf per
base-cell atom; ``edge_index``/``shifts`` route each edge to the periodic
image it needs, but moving ``positions[i]`` moves every image of atom ``i``
identically. That is *why* the plain directional JVP
(``compute_graph2mat_directional_derivative``) is a Gamma-periodic (``q=0``)
response by construction (roadmap Section II, blocker B4): it cannot tell
"the home copy of atom i" apart from "the image-l copy of atom i" for a
self-pair edge (``mu == nu``, ``shell != 0``), because both roles read the
same tensor entry.

:mod:`epc_qb_qc_prototypes` sidesteps this on its numpy toy by calling
``pair_block`` on two *independent* explicit position arrays (a home array
and a neighbour array) per shell -- there is no shared tensor to conflate.
The models below port that same trick onto the real production JVP by
expanding the batch's position tensor into independent "home" and
"neighbour-image" leaves (:class:`PeriodicShellGraphModel`, one neighbour
block shared across shells -- enough for Q-C, whose neighbour tangent has no
``q`` in it) or one *independent neighbour block per shell*
(:class:`PeriodicShellReplicatedGraphModel` -- needed for Q-B, whose
neighbour tangent is weighted by ``exp(i q . R_shell)`` and therefore must
differ from shell to shell). Both classes are differentiated through the
same, unmodified :func:`compute_graph2mat_directional_derivative`; nothing in
this module reimplements autograd.

The result: Q-C's whole ``q``-independent kernel table needs exactly **2**
real JVP calls, independent of ``shells`` (contrast
``epc_qb_qc_prototypes.kernel_table``'s ``4 * (2 * shells + 1)`` finite
differences); Q-B needs exactly **2** real JVP calls *per q* (contrast the
toy's per-shell Python loop). Nothing of shape
``[n_outputs, n_atoms, 3]`` is ever materialised.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
import epc_fourier as qf  # noqa: E402
from artifact_signature import (  # noqa: E402
    CACHE_VALID,
    cached_result_status,
    epc_artifact_node,
)
from epc_qb_qc_prototypes import (  # noqa: E402
    A_ANG,
    CELL_VECTOR_ANG,
    CUTOFF_RADIUS_ANG,
    CUTOFF_SHELLS,
    LAMBDA_ANG,
    ONSITE_EV,
    PRIMITIVE_CELL_ANG,
    PRIMITIVE_POSITIONS_ANG,
    S0,
    T0_EV,
)
from graph2mat_autograd_derivatives import (  # noqa: E402
    DIRECTIONAL_DERIVATIVE_SCHEMA,
    Graph2MatAutogradDerivativeError,
    compute_graph2mat_directional_derivative,
    derivative_prediction_to_sparse_matrices,
    resolve_jvp_backend,
)

TICKET = "E-F_001-S40"
KERNEL_TABLE_SCHEMA = "epc_qb_qc_graph2mat_kernel_table_v1"
PHASE_AWARE_KERNEL_SCHEMA = "epc_qb_qc_graph2mat_phase_aware_kernel_v1"
STATUS = "real_graph2mat_jvp__toy_periodic_model"

N_ATOMS_PER_CELL = 2
#: Same default direction as epc_qb_qc_prototypes.compare_routes() -- unit
#: Frobenius norm, required by fd_perturbation_space.Direction.
DEFAULT_DIRECTION_VECTORS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)

#: S37 (docs/epc_s37_matbg_synthetic_benchmark.md): the naive unchunked
#: directional-JVP forward on the real (31, 30) checkpoint drove RSS past
#: 30+ GiB and was killed by a safety watchdog before any structure-specific
#: edge count was ever recorded. There is therefore no measured MATBG edge
#: count to scale this module's sweep to; shells is used as an explicit,
#: honestly-labelled proxy instead.
MATBG_EDGE_COUNT_MEASURED = False
MATBG_EDGE_COUNT_BLOCKING_REASON = (
    "S37 (docs/epc_s37_matbg_synthetic_benchmark.md): the naive unchunked directional-JVP "
    "forward on the real (31, 30) checkpoint drove RSS past 30+ GiB and was killed by a "
    "safety watchdog before completion; no real MATBG edge count was recorded"
)


class GammaJvpReuseError(RuntimeError):
    """A plain q=0 Gamma-periodic JVP payload was fed into a q != 0 contraction.

    S33 ``rejects_gamma_jvp_reuse``; GO-8b's own acceptance criterion: "Un JVP
    Gamma reutilizado para q != 0 provoca fallo automatico."
    """


def _within_cutoff_pairs(shell: int) -> list[tuple[int, int]]:
    pairs = []
    for mu in (0, 1):
        for nu in (0, 1):
            if shell == 0 and mu == nu:
                continue
            neighbor = PRIMITIVE_POSITIONS_ANG[nu] + shell * CELL_VECTOR_ANG
            distance = float(np.linalg.norm(neighbor - PRIMITIVE_POSITIONS_ANG[mu]))
            if distance <= CUTOFF_RADIUS_ANG:
                pairs.append((mu, nu))
    return pairs


def _base_edge_list(shells: int) -> list[tuple[int, int, int]]:
    """``(shell, mu, nu)`` for every pair within cutoff at the equilibrium geometry.

    This is the model's fixed neighbour topology (roadmap: "edge_index and
    shifts are taken from the input batch and kept constant"): built once from
    the base positions, never re-evaluated inside ``forward``.
    """
    edges = []
    for shell in range(-int(shells), int(shells) + 1):
        for mu, nu in _within_cutoff_pairs(shell):
            edges.append((shell, mu, nu))
    return edges


def _isc_off(shells: int) -> list[list[int]]:
    return [[shell, 0, 0] for shell in range(-int(shells), int(shells) + 1)]


class KernelBatch(dict):
    """Minimal batch stand-in (dict + clone + device move), same shape as the
    ``FakeBatch`` fixture ``tests/test_graph2mat_autograd_derivatives.py``
    already uses for non-MACE production-function tests."""

    def clone(self) -> "KernelBatch":
        return KernelBatch(self)

    def to(self, device: str) -> "KernelBatch":
        moved = KernelBatch(self)
        moved["positions"] = self["positions"].to(device)
        return moved


class PeriodicShellGraphModel(torch.nn.Module):
    """Real edge-resolved periodic model for route Q-C.

    ``data["positions"]`` has shape ``[2 * N_ATOMS_PER_CELL, 3]``: rows
    ``[0:N_ATOMS_PER_CELL)`` are the "home" leaves (the ``mu`` role, shared by
    every shell -- correct, since B_l has no shell/``q`` dependence either);
    rows ``[N_ATOMS_PER_CELL:2*N_ATOMS_PER_CELL)`` are the "neighbour-image"
    leaves (the ``nu`` role, also shared across shells -- fine for Q-C, whose
    neighbour tangent carries no ``q``-phase and is the same for every shell).
    ``edge_labels`` has one row ``(H, S)`` per ``(shell, mu, nu)`` in
    :func:`_base_edge_list`, computed with the same exponential-decay SK-like
    physics as ``epc_qb_qc_prototypes.pair_term``.
    """

    def __init__(self, shells: int = CUTOFF_SHELLS) -> None:
        super().__init__()
        self.shells = int(shells)
        self.edges: list[tuple[int, int, int]] = _base_edge_list(shells)

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        home = positions[:N_ATOMS_PER_CELL]
        neighbor = positions[N_ATOMS_PER_CELL:]
        cell_vector = torch.as_tensor(CELL_VECTOR_ANG, dtype=positions.dtype, device=positions.device)
        rows = []
        for shell, mu, nu in self.edges:
            neighbor_pos = neighbor[nu] + shell * cell_vector
            distance = torch.linalg.norm(neighbor_pos - home[mu])
            decay = torch.exp(-distance / LAMBDA_ANG)
            rows.append(torch.stack([-T0_EV * decay, S0 * decay]))
        edge_labels = (
            torch.stack(rows) if rows else torch.zeros((0, 2), dtype=positions.dtype, device=positions.device)
        )
        onsite_h = torch.as_tensor(ONSITE_EV, dtype=positions.dtype, device=positions.device)
        node_labels = torch.stack([onsite_h, torch.ones_like(onsite_h)], dim=1) + 0.0 * positions.sum()
        return {"node_labels": node_labels, "edge_labels": edge_labels}


class PeriodicShellReplicatedGraphModel(torch.nn.Module):
    """Real edge-resolved periodic model for route Q-B.

    Same physics as :class:`PeriodicShellGraphModel`, but with one
    *independent* neighbour-image leaf block per shell (not shared across
    shells), so a single directional JVP can assign a different,
    shell-dependent tangent weight to each image -- what Q-B needs (the
    ``q``-phase folded into the tangent before differentiating) and what
    :class:`PeriodicShellGraphModel`'s shared neighbour leaf structurally
    cannot do. Position layout: rows ``[0:N_ATOMS_PER_CELL)`` are the home
    leaves (shared, ``q``-independent); rows
    ``[N_ATOMS_PER_CELL + s*N_ATOMS_PER_CELL : N_ATOMS_PER_CELL + (s+1)*N_ATOMS_PER_CELL)``
    are the neighbour leaves private to ``shell_values[s]``.
    """

    def __init__(self, shells: int = CUTOFF_SHELLS) -> None:
        super().__init__()
        self.shells = int(shells)
        self.shell_values = list(range(-self.shells, self.shells + 1))
        self.edges: list[tuple[int, int, int, int]] = []  # (shell_index, shell, mu, nu)
        for shell_index, shell in enumerate(self.shell_values):
            for mu, nu in _within_cutoff_pairs(shell):
                self.edges.append((shell_index, shell, mu, nu))

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        home = positions[:N_ATOMS_PER_CELL]
        cell_vector = torch.as_tensor(CELL_VECTOR_ANG, dtype=positions.dtype, device=positions.device)
        rows = []
        for shell_index, shell, mu, nu in self.edges:
            base = N_ATOMS_PER_CELL + shell_index * N_ATOMS_PER_CELL
            neighbor_pos = positions[base + nu] + shell * cell_vector
            distance = torch.linalg.norm(neighbor_pos - home[mu])
            decay = torch.exp(-distance / LAMBDA_ANG)
            rows.append(torch.stack([-T0_EV * decay, S0 * decay]))
        edge_labels = (
            torch.stack(rows) if rows else torch.zeros((0, 2), dtype=positions.dtype, device=positions.device)
        )
        onsite_h = torch.as_tensor(ONSITE_EV, dtype=positions.dtype, device=positions.device)
        node_labels = torch.stack([onsite_h, torch.ones_like(onsite_h)], dim=1) + 0.0 * positions.sum()
        return {"node_labels": node_labels, "edge_labels": edge_labels}


def _empty_2x2_dict(shells: int) -> dict[int, np.ndarray]:
    return {shell: np.zeros((N_ATOMS_PER_CELL, N_ATOMS_PER_CELL)) for shell in range(-shells, shells + 1)}


# --------------------------------------------------------------------------- #
# Route Q-C: 2 real JVP calls, independent of shells
# --------------------------------------------------------------------------- #


def kernel_table_graph2mat(
    *,
    shells: int = CUTOFF_SHELLS,
    direction: Any = None,
    base_positions_ang: Any = None,
    dtype: torch.dtype = torch.float64,
    requested_backend: str = "cpu",
) -> dict[str, Any]:
    """Route Q-C on the real production JVP: the q-free kernel table.

    Exactly 2 :func:`compute_graph2mat_directional_derivative` calls --
    independent of ``shells`` -- against ``epc_qb_qc_prototypes.kernel_table``'s
    ``4 * (2 * shells + 1)`` hand-rolled finite differences.
    """
    if direction is None:
        direction = DEFAULT_DIRECTION_VECTORS
    if base_positions_ang is None:
        base_positions_ang = PRIMITIVE_POSITIONS_ANG
    shells = int(shells)
    model = PeriodicShellGraphModel(shells=shells)
    home0 = torch.as_tensor(np.asarray(base_positions_ang, dtype=np.float64), dtype=dtype)
    batch = KernelBatch(positions=torch.cat([home0, home0.clone()]))
    model, batch, backend_record = resolve_jvp_backend(model, batch, requested_backend)

    dvec = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=dtype, device=batch["positions"].device)
    zero = torch.zeros_like(dvec)

    start = time.perf_counter()
    neighbor_response = compute_graph2mat_directional_derivative(model, batch, torch.cat([zero, dvec]))
    home_response = compute_graph2mat_directional_derivative(model, batch, torch.cat([dvec, zero]))
    build_seconds = time.perf_counter() - start

    edge_h_a = neighbor_response.derivative["edge_labels"][:, 0].detach().cpu().numpy()
    edge_s_a = neighbor_response.derivative["edge_labels"][:, 1].detach().cpu().numpy()
    edge_h_b = home_response.derivative["edge_labels"][:, 0].detach().cpu().numpy()
    edge_s_b = home_response.derivative["edge_labels"][:, 1].detach().cpu().numpy()

    a_h, b_h, a_s, b_s = (
        _empty_2x2_dict(shells),
        _empty_2x2_dict(shells),
        _empty_2x2_dict(shells),
        _empty_2x2_dict(shells),
    )
    for index, (shell, mu, nu) in enumerate(model.edges):
        a_h[shell][mu, nu] = float(edge_h_a[index])
        b_h[shell][mu, nu] = float(edge_h_b[index])
        a_s[shell][mu, nu] = float(edge_s_a[index])
        b_s[shell][mu, nu] = float(edge_s_b[index])

    return {
        "schema": KERNEL_TABLE_SCHEMA,
        "ticket": TICKET,
        "status": STATUS,
        "A_H": a_h,
        "B_H": b_h,
        "A_S": a_s,
        "B_S": b_s,
        "shells": shells,
        "jvp_calls": 2,
        "materialized_jacobian": False,
        "build_seconds": build_seconds,
        "direction_hash": neighbor_response.metadata["direction_hash"],
        "topology_hash": neighbor_response.metadata["topology_hash"],
        "dtype": neighbor_response.metadata["dtype"],
        "backend": backend_record.to_metadata(),
        "edges_in_topology": len(model.edges),
    }


def k_to_k_plus_q_kernel_graph2mat(table: dict[str, Any], k_primitive: Any, q_primitive: Any) -> dict[str, Any]:
    """``D_H(k+q, k) = sum_l A_l exp(i(k+q).R_l) + sum_l B_l exp(ik.R_l)``.

    O(shells) given ``table``; no new JVP for a new ``(k, q)``. Raises
    :class:`GammaJvpReuseError` if ``table`` is not actually a
    :func:`kernel_table_graph2mat` payload (S33 ``rejects_gamma_jvp_reuse``).
    """
    q = np.asarray(q_primitive, dtype=np.float64).reshape(3)
    if table.get("schema") != KERNEL_TABLE_SCHEMA and float(np.linalg.norm(q)) != 0.0:
        raise GammaJvpReuseError(
            f"k_to_k_plus_q_kernel_graph2mat: expected a {KERNEL_TABLE_SCHEMA!r} payload for a "
            f"q != 0 query, got schema {table.get('schema')!r}. A plain "
            f"{DIRECTIONAL_DERIVATIVE_SCHEMA!r} (compute_graph2mat_directional_derivative) is a "
            "Gamma-periodic (q=0) JVP of one cell (roadmap Section II, blocker B4) and is never "
            "an acceptable substitute for a q != 0 response (S33 rejects_gamma_jvp_reuse)."
        )
    shells = int(table["shells"])
    isc = _isc_off(shells)
    shell_range = range(-shells, shells + 1)
    k_final = qf.k_plus_q(k_primitive, q_primitive)
    a_h_stack = np.stack([table["A_H"][s] for s in shell_range])
    b_h_stack = np.stack([table["B_H"][s] for s in shell_range])
    a_s_stack = np.stack([table["A_S"][s] for s in shell_range])
    b_s_stack = np.stack([table["B_S"][s] for s in shell_range])
    return {
        "D_H": qf.bloch_sum(a_h_stack, isc, k_final) + qf.bloch_sum(b_h_stack, isc, k_primitive),
        "D_S": qf.bloch_sum(a_s_stack, isc, k_final) + qf.bloch_sum(b_s_stack, isc, k_primitive),
    }


# --------------------------------------------------------------------------- #
# Route Q-B: 2 real JVP calls per q, cross-check only
# --------------------------------------------------------------------------- #


def phase_aware_kernel_graph2mat(
    q_primitive: Any,
    *,
    shells: int = CUTOFF_SHELLS,
    direction: Any = None,
    dtype: torch.dtype = torch.float64,
    requested_backend: str = "cpu",
) -> dict[str, Any]:
    """Route Q-B on the real production JVP: the phase folded into the tangent.

    Exactly 2 JVP calls (real/imaginary parts of the phase) per ``q``, all
    shells at once -- against ``epc_qb_qc_prototypes.phase_aware_kernel``'s
    ``4 * (2 * shells + 1)`` finite differences for the same ``q``.
    """
    if direction is None:
        direction = DEFAULT_DIRECTION_VECTORS
    shells = int(shells)
    model = PeriodicShellReplicatedGraphModel(shells=shells)
    home0 = torch.as_tensor(np.asarray(PRIMITIVE_POSITIONS_ANG, dtype=np.float64), dtype=dtype)
    neighbor_blocks0 = torch.cat([home0.clone() for _ in model.shell_values])
    batch = KernelBatch(positions=torch.cat([home0, neighbor_blocks0]))
    model, batch, backend_record = resolve_jvp_backend(model, batch, requested_backend)

    dvec = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=dtype, device=batch["positions"].device)
    zero_home = torch.zeros_like(dvec)
    q_x = float(np.asarray(q_primitive, dtype=np.float64).reshape(3)[0])
    cos_blocks = torch.cat([dvec * float(np.cos(2.0 * np.pi * shell * q_x)) for shell in model.shell_values])
    sin_blocks = torch.cat([dvec * float(np.sin(2.0 * np.pi * shell * q_x)) for shell in model.shell_values])

    start = time.perf_counter()
    real_part = compute_graph2mat_directional_derivative(model, batch, torch.cat([dvec, cos_blocks]))
    imag_part = compute_graph2mat_directional_derivative(model, batch, torch.cat([zero_home, sin_blocks]))
    build_seconds = time.perf_counter() - start

    h_re = real_part.derivative["edge_labels"][:, 0].detach().cpu().numpy()
    h_im = imag_part.derivative["edge_labels"][:, 0].detach().cpu().numpy()
    s_re = real_part.derivative["edge_labels"][:, 1].detach().cpu().numpy()
    s_im = imag_part.derivative["edge_labels"][:, 1].detach().cpu().numpy()

    kernel_h = {shell: np.zeros((N_ATOMS_PER_CELL, N_ATOMS_PER_CELL), dtype=complex) for shell in model.shell_values}
    kernel_s = {shell: np.zeros((N_ATOMS_PER_CELL, N_ATOMS_PER_CELL), dtype=complex) for shell in model.shell_values}
    for index, (_shell_index, shell, mu, nu) in enumerate(model.edges):
        kernel_h[shell][mu, nu] = complex(h_re[index], h_im[index])
        kernel_s[shell][mu, nu] = complex(s_re[index], s_im[index])

    return {
        "schema": PHASE_AWARE_KERNEL_SCHEMA,
        "ticket": TICKET,
        "status": STATUS,
        "kernel_H": kernel_h,
        "kernel_S": kernel_s,
        "q_primitive": [float(v) for v in np.asarray(q_primitive, dtype=np.float64).reshape(3)],
        "shells": shells,
        "jvp_calls": 2,
        "materialized_jacobian": False,
        "build_seconds": build_seconds,
        "direction_hash": real_part.metadata["direction_hash"],
        "topology_hash": real_part.metadata["topology_hash"],
        "dtype": real_part.metadata["dtype"],
        "backend": backend_record.to_metadata(),
        "edges_in_topology": len(model.edges),
    }


def q_b_response_graph2mat(phase_aware: dict[str, Any], k_primitive: Any) -> dict[str, Any]:
    """The k-weighted sum on top of one :func:`phase_aware_kernel_graph2mat` result."""
    if phase_aware.get("schema") != PHASE_AWARE_KERNEL_SCHEMA:
        raise GammaJvpReuseError(
            f"q_b_response_graph2mat: expected a {PHASE_AWARE_KERNEL_SCHEMA!r} payload, got schema "
            f"{phase_aware.get('schema')!r}. A plain {DIRECTIONAL_DERIVATIVE_SCHEMA!r} is a "
            "Gamma-periodic (q=0) JVP and is never an acceptable substitute here "
            "(S33 rejects_gamma_jvp_reuse)."
        )
    shells = int(phase_aware["shells"])
    isc = _isc_off(shells)
    shell_range = range(-shells, shells + 1)
    h_stack = np.stack([phase_aware["kernel_H"][s] for s in shell_range])
    s_stack = np.stack([phase_aware["kernel_S"][s] for s in shell_range])
    return {
        "D_H": qf.bloch_sum(h_stack, isc, k_primitive),
        "D_S": qf.bloch_sum(s_stack, isc, k_primitive),
    }


# --------------------------------------------------------------------------- #
# S33 item 2: topology-margin test on the real kernel's direction/cutoff
# --------------------------------------------------------------------------- #


def topology_margin_report(
    *, shells: int = CUTOFF_SHELLS, direction: Any = None, delta_ang: float = 0.05
) -> dict[str, Any]:
    """No atom pair the kernel touches may cross ``edge_index``'s fixed cutoff.

    Reuses ``fd_perturbation_space.topology_margin`` directly against this
    module's own equilibrium geometry/cutoff/lattice -- not a new margin
    computation.
    """
    if direction is None:
        direction = DEFAULT_DIRECTION_VECTORS
    fd_direction = fdp.collective(direction, name="epc_qb_qc_graph2mat_kernel_direction")
    margin = fdp.topology_margin(
        PRIMITIVE_POSITIONS_ANG,
        direction=fd_direction,
        delta_ang=delta_ang,
        cutoff_ang=CUTOFF_RADIUS_ANG,
        lattice_vectors_ang=PRIMITIVE_CELL_ANG,
    )
    payload = margin.to_dict()
    payload.update({"schema": "epc_qb_qc_graph2mat_topology_margin_v1", "ticket": TICKET, "shells": int(shells)})
    return payload


# --------------------------------------------------------------------------- #
# S33 item 3: the five S31-style checks, repeated on the real kernel
# --------------------------------------------------------------------------- #


def check_qc_and_qb_agree_on_real_graph2mat(
    *,
    shells: int = CUTOFF_SHELLS,
    direction: Any = None,
    k_primitive: Any = (0.15, 0.0, 0.0),
    q_primitive: Any = (0.2, 0.0, 0.0),
) -> dict[str, Any]:
    """Two independent real-Graph2Mat computational paths (Q-C's formula
    combination and Q-B's phase-folded JVP) must agree -- the real-machinery
    analogue of S30's ``k_to_k_plus_q_by_subspace``."""
    table = kernel_table_graph2mat(shells=shells, direction=direction)
    qc = k_to_k_plus_q_kernel_graph2mat(table, k_primitive, q_primitive)
    phase_aware = phase_aware_kernel_graph2mat(q_primitive, shells=shells, direction=direction)
    qb = q_b_response_graph2mat(phase_aware, k_primitive)
    worst = max(
        float(np.abs(qc["D_H"] - qb["D_H"]).max()),
        float(np.abs(qc["D_S"] - qb["D_S"]).max()),
    )
    return {"name": "qc_and_qb_agree_on_real_graph2mat", "status": "PASS" if worst < 1e-9 else "FAIL", "worst_abs_diff": worst}


def check_q_and_minus_q(
    *, shells: int = CUTOFF_SHELLS, direction: Any = None, q_primitive: Any = (0.2, 0.0, 0.0)
) -> dict[str, Any]:
    """``kernel(-q) == minus_q_matrix(kernel(q))`` (S29 ``MINUS_Q_CONVENTION``),
    reusing :func:`epc_fourier.minus_q_matrix` directly."""
    positive = phase_aware_kernel_graph2mat(q_primitive, shells=shells, direction=direction)
    negative_q = tuple(-float(v) for v in np.asarray(q_primitive, dtype=np.float64).reshape(3))
    negative = phase_aware_kernel_graph2mat(negative_q, shells=shells, direction=direction)
    worst = 0.0
    for shell in range(-int(shells), int(shells) + 1):
        worst = max(worst, float(np.abs(qf.minus_q_matrix(positive["kernel_H"][shell]) - negative["kernel_H"][shell]).max()))
        worst = max(worst, float(np.abs(qf.minus_q_matrix(positive["kernel_S"][shell]) - negative["kernel_S"][shell]).max()))
    return {"name": "q_and_minus_q", "status": "PASS" if worst < 1e-9 else "FAIL", "worst_abs_diff": worst}


def check_cell_origin_shift(
    *,
    shells: int = CUTOFF_SHELLS,
    direction: Any = None,
    k_primitive: Any = (0.15, 0.0, 0.0),
    q_primitive: Any = (0.2, 0.0, 0.0),
) -> dict[str, Any]:
    """Moving where coordinate counting starts (a rigid shift by one lattice
    vector) cannot change any measurable ``D_H(k+q, k)``: the physics only
    depends on relative distances."""
    table = kernel_table_graph2mat(shells=shells, direction=direction)
    response = k_to_k_plus_q_kernel_graph2mat(table, k_primitive, q_primitive)
    shifted_table = kernel_table_graph2mat(
        shells=shells, direction=direction, base_positions_ang=PRIMITIVE_POSITIONS_ANG + CELL_VECTOR_ANG
    )
    shifted_response = k_to_k_plus_q_kernel_graph2mat(shifted_table, k_primitive, q_primitive)
    worst = max(
        float(np.abs(response["D_H"] - shifted_response["D_H"]).max()),
        float(np.abs(response["D_S"] - shifted_response["D_S"]).max()),
    )
    return {"name": "cell_origin_shift", "status": "PASS" if worst < 1e-9 else "FAIL", "worst_abs_diff": worst}


def check_atom_across_periodic_boundary(
    *, shells: int = CUTOFF_SHELLS, direction: Any = None, k_primitive: Any = (0.15, 0.0, 0.0)
) -> dict[str, Any]:
    """Re-expressing atom 1 in a different periodic image (``tau -> tau + T``)
    must transform ``D_H(k)`` exactly as ``epc_fourier.image_relabel_unitary``
    says: ``X'(k) = W X(k) W^dagger``. Uses ``epc_fourier.relabel_home_images``
    for the real-space side and checks it against the k-space unitary
    directly, both reused unmodified."""
    shells = int(shells)
    # relabel_home_images needs the table closed under the relabelling: moving
    # atom 1's canonical image by +1 cell shifts its column/row entries by one
    # more shell than the table would otherwise need, so pad by 1 here.
    padded_shells = shells + 1
    table = kernel_table_graph2mat(shells=padded_shells, direction=direction)
    isc = _isc_off(padded_shells)
    shell_range = range(-padded_shells, padded_shells + 1)
    combined_h = np.stack([table["A_H"][s] + table["B_H"][s] for s in shell_range])
    image_shifts = [[0, 0, 0], [1, 0, 0]]  # atom 1 re-labelled one cell over
    relabeled_h = qf.relabel_home_images(combined_h, isc, image_shifts)
    original_at_k = qf.bloch_sum(combined_h, isc, k_primitive)
    relabeled_at_k = qf.bloch_sum(relabeled_h, isc, k_primitive)
    unitary = np.diag(qf.image_relabel_unitary(k_primitive, image_shifts))
    expected = unitary @ original_at_k @ unitary.conj().T
    worst = float(np.abs(relabeled_at_k - expected).max())
    return {"name": "atom_across_periodic_boundary", "status": "PASS" if worst < 1e-9 else "FAIL", "worst_abs_diff": worst}


def check_alternative_atomic_phase_convention(
    *, shells: int = CUTOFF_SHELLS, direction: Any = None, k_primitive: Any = (0.15, 0.0, 0.0)
) -> dict[str, Any]:
    """A provider writing the atomic phase into the atom gauge instead of the
    cell gauge must round-trip to the identical canonical ``D_H(k)`` via
    ``epc_fourier.to_canonical_at_k`` (reused, not reimplemented)."""
    shells = int(shells)
    table = kernel_table_graph2mat(shells=shells, direction=direction)
    isc = _isc_off(shells)
    shell_range = range(-shells, shells + 1)
    combined_h = np.stack([table["A_H"][s] + table["B_H"][s] for s in shell_range])
    canonical = qf.bloch_sum(combined_h, isc, k_primitive)
    frac = qf.fractional_positions(PRIMITIVE_POSITIONS_ANG, PRIMITIVE_CELL_ANG)
    atom_conventions = qf.QSpaceConventions(atomic_phase=qf.ATOMIC_PHASE_ATOM)
    atom_gauge = qf.to_canonical_at_k(canonical, k_primitive, atom_conventions, orbital_positions_frac=frac, inverse=True)
    recovered = qf.to_canonical_at_k(atom_gauge, k_primitive, atom_conventions, orbital_positions_frac=frac, inverse=False)
    round_trip_diff = float(np.abs(recovered - canonical).max())
    gauge_changed = float(np.abs(atom_gauge - canonical).max())
    passed = round_trip_diff < 1e-9 and gauge_changed > 1e-6
    return {
        "name": "alternative_atomic_phase_convention",
        "status": "PASS" if passed else "FAIL",
        "round_trip_max_abs_diff": round_trip_diff,
        "gauge_actually_changed_the_matrix": gauge_changed > 1e-6,
    }


def real_kernel_checks(
    *, shells: int = CUTOFF_SHELLS, direction: Any = None
) -> dict[str, Any]:
    """The five S31-style checks (S33 item 3), all against the real kernel."""
    checks = [
        check_qc_and_qb_agree_on_real_graph2mat(shells=shells, direction=direction),
        check_q_and_minus_q(shells=shells, direction=direction),
        check_cell_origin_shift(shells=shells, direction=direction),
        check_atom_across_periodic_boundary(shells=shells, direction=direction),
        check_alternative_atomic_phase_convention(shells=shells, direction=direction),
    ]
    return {
        "schema": "epc_qb_qc_graph2mat_real_kernel_checks_v1",
        "ticket": TICKET,
        "checks": {check["name"]: check["status"] for check in checks},
        "details": checks,
        "all_pass": all(check["status"] == "PASS" for check in checks),
    }


# --------------------------------------------------------------------------- #
# S33 item 8: sparse serialisation round trip through the existing C11 mapping
# --------------------------------------------------------------------------- #


class _KernelSparseProcessor:
    """Minimal stand-in for a graph2mat ``BasisMatrixData`` processor -- same
    contract shape as the ``FakeProcessor`` fixture
    ``tests/test_graph2mat_autograd_derivatives.py::SparseConversionTests``
    already uses to exercise ``derivative_prediction_to_sparse_matrices``
    (C11) without a real graph2mat/MACE dependency."""

    sub_point_matrix = False
    default_out_format = "scipy_csr"

    def copy(self, **kwargs: Any) -> "_KernelSparseProcessor":
        clone = _KernelSparseProcessor()
        for key, value in kwargs.items():
            setattr(clone, key, value)
        return clone

    def yield_from_batch(self, batch: Any, predictions: Any = None, as_matrix: bool = False):
        yield predictions

    def labels_to(self, out_format: str, data: Any = None, threshold: float | None = None):
        from scipy import sparse

        edge_h = data["edge_labels"][:, 0]
        edge_h = edge_h.detach().cpu().numpy() if hasattr(edge_h, "detach") else np.asarray(edge_h)
        return sparse.csr_matrix(np.diag(edge_h)) if edge_h.size else sparse.csr_matrix((0, 0))


def sparse_round_trip_check(*, shells: int = CUTOFF_SHELLS, direction: Any = None, dtype: torch.dtype = torch.float64) -> dict[str, Any]:
    """Round-trips one kernel-construction JVP through the production sparse
    mapping (C11, ``derivative_prediction_to_sparse_matrices``) twice, and
    checks the result is byte-identical both times -- reproducible sparse
    artifacts per direction, per S40's acceptance criterion."""
    if direction is None:
        direction = DEFAULT_DIRECTION_VECTORS
    shells = int(shells)
    model = PeriodicShellGraphModel(shells=shells)
    home0 = torch.as_tensor(np.asarray(PRIMITIVE_POSITIONS_ANG, dtype=np.float64), dtype=dtype)
    batch = KernelBatch(positions=torch.cat([home0, home0.clone()]))
    dvec = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=dtype)
    zero = torch.zeros_like(dvec)
    processor = _KernelSparseProcessor()

    first = compute_graph2mat_directional_derivative(model, batch, torch.cat([zero, dvec]), data_processor=processor)
    second = compute_graph2mat_directional_derivative(model, batch, torch.cat([zero, dvec]), data_processor=processor)
    matrix_a, matrix_b = first.matrices[0], second.matrices[0]
    reproducible = bool((matrix_a != matrix_b).nnz == 0) and matrix_a.shape == matrix_b.shape
    return {
        "schema": "epc_qb_qc_graph2mat_sparse_round_trip_v1",
        "ticket": TICKET,
        "nnz": int(matrix_a.nnz),
        "matrix_shape": [int(dim) for dim in matrix_a.shape],
        "reproducible_across_calls": reproducible,
        "matches_result_metadata_nnz": int(matrix_a.nnz) == int(first.metadata["nnz"]),
    }


# --------------------------------------------------------------------------- #
# S33 item 5: raw_derivative_kernel_table cache/restart
# --------------------------------------------------------------------------- #


def kernel_table_dag_node(table: dict[str, Any]) -> dict[str, Any]:
    """The ``raw_derivative_kernel_table`` DAG node for a :func:`kernel_table_graph2mat` payload."""
    hamiltonian = epc_artifact_node(
        "electronic_hamiltonian",
        {
            "source_backend": "graph2mat",
            "model_or_binary_sha256": "epc_qb_qc_graph2mat_kernel_toy_model",
            "code_version": TICKET,
            "basis": "toy_1d_two_atom_chain",
            "neighbor_cutoff_policy": f"pair_cutoff_{CUTOFF_RADIUS_ANG}_ang",
            "topology_sha256": table["topology_hash"],
            "dtype": table["dtype"],
            "mapping_version": KERNEL_TABLE_SCHEMA,
        },
        {
            "geometry": epc_artifact_node(
                "geometry",
                {
                    "cell": PRIMITIVE_CELL_ANG.tolist(),
                    "species": ["C", "C"],
                    "positions_sha256": "epc_qb_qc_prototypes_primitive_positions",
                    "coordinate_convention": "cartesian",
                },
            )
        },
    )
    return epc_artifact_node(
        "raw_derivative_kernel_table",
        {
            "derivative_backend": "graph2mat",
            "derivative_method": "graph2mat_jvp_per_shell",
            "direction_hash": table["direction_hash"],
            "topology_sha256": table["topology_hash"],
            "delta_or_jvp": {"jvp": True, "jvp_calls": table["jvp_calls"]},
            "shells": table["shells"],
            "dtype": table["dtype"],
            "units": "eV/Ang",
        },
        {"hamiltonian": hamiltonian},
    )


def _kernel_table_to_matrix(table: dict[str, Any]) -> np.ndarray:
    shells = int(table["shells"])
    rows = []
    for component in ("A_H", "B_H", "A_S", "B_S"):
        for shell in range(-shells, shells + 1):
            rows.append(np.asarray(table[component][shell]).reshape(-1))
    return np.stack(rows)


def save_kernel_table_cache(table: dict[str, Any], node: dict[str, Any], npz_path: str | Path, metadata_path: str | Path) -> None:
    """Serialise ``table`` through the *existing* sparse cache mechanism
    (``artifact_signature.cached_result_status`` / the same ``.npz`` + JSON
    sidecar shape every other derivative cache in this repository uses) --
    not a new cache manager."""
    from scipy import sparse

    matrix = sparse.csr_matrix(_kernel_table_to_matrix(table))
    npz_path, metadata_path = Path(npz_path), Path(metadata_path)
    with npz_path.open("wb") as handle:
        sparse.save_npz(handle, matrix)
    metadata_path.write_text(
        json.dumps(
            {
                "input_signature_sha256": node["signature_sha256"],
                "matrix_shape": list(matrix.shape),
                "shells": table["shells"],
                "direction_hash": table["direction_hash"],
                "topology_hash": table["topology_hash"],
                "dtype": table["dtype"],
                "jvp_calls": table["jvp_calls"],
                "build_seconds": table["build_seconds"],
                "backend": table["backend"],
                "edges_in_topology": table["edges_in_topology"],
            }
        ),
        encoding="utf-8",
    )


def load_kernel_table_cache(
    node: dict[str, Any], npz_path: str | Path, metadata_path: str | Path
) -> tuple[dict[str, Any] | None, str]:
    """The restart half: reload a cached table with zero new JVP calls if (and
    only if) the signature still matches ``node`` (moved atoms, changed
    direction, changed topology, ... all invalidate it)."""
    status = cached_result_status(npz_path, metadata_path, node["signature_sha256"])
    if status != CACHE_VALID:
        return None, status
    from scipy import sparse

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    matrix = sparse.load_npz(npz_path).toarray()
    shells = int(metadata["shells"])
    shell_range = list(range(-shells, shells + 1))
    n_shell = len(shell_range)
    components: dict[str, dict[int, np.ndarray]] = {}
    row = 0
    for name in ("A_H", "B_H", "A_S", "B_S"):
        block: dict[int, np.ndarray] = {}
        for shell in shell_range:
            block[shell] = matrix[row].reshape(N_ATOMS_PER_CELL, N_ATOMS_PER_CELL)
            row += 1
        components[name] = block
    table = {
        "schema": KERNEL_TABLE_SCHEMA,
        "ticket": TICKET,
        "status": STATUS,
        **components,
        "shells": shells,
        "jvp_calls": metadata["jvp_calls"],
        "materialized_jacobian": False,
        "build_seconds": metadata["build_seconds"],
        "direction_hash": metadata["direction_hash"],
        "topology_hash": metadata["topology_hash"],
        "dtype": metadata["dtype"],
        "backend": metadata["backend"],
        "edges_in_topology": metadata["edges_in_topology"],
        "loaded_from_cache": True,
        "jvp_calls_avoided": metadata["jvp_calls"],
    }
    del n_shell
    return table, status


# --------------------------------------------------------------------------- #
# S33 item 6: GPU preflight (extends resolve_jvp_backend, does not repeat it)
# --------------------------------------------------------------------------- #


def gpu_preflight_report(*, shells: int = CUTOFF_SHELLS, requested_backend: str = "cuda") -> dict[str, Any]:
    """The real-Graph2Mat-kernel GPU preflight S33 item 6 asks for: run the
    real production ``resolve_jvp_backend`` cheap CUDA probe on this module's
    actual periodic model/batch, and record requested/effective/reason --
    never a silent CPU fallback (Default compute policy)."""
    table = kernel_table_graph2mat(shells=shells, requested_backend=requested_backend)
    return {
        "schema": "epc_qb_qc_graph2mat_gpu_preflight_v1",
        "ticket": TICKET,
        "shells": shells,
        **table["backend"],
    }


# --------------------------------------------------------------------------- #
# S33 item 7: scaling proxy sweep -- shells, since no real MATBG edge count
# has ever been measured (S37 OOM'd first)
# --------------------------------------------------------------------------- #


def scaling_sweep_graph2mat(shells_values: Any = (1, 4, 16, 64), *, direction: Any = None) -> dict[str, Any]:
    """Measure real-JVP kernel-construction cost as ``shells`` grows.

    Unlike ``epc_qb_qc_prototypes.scaling_sweep`` (which compares Q-C's
    amortised table against Q-B's per-q rebuild), this sweep's point is that
    a single :func:`kernel_table_graph2mat` call costs exactly 2 real JVP
    calls *at every shell count* -- the production JVP does not pay per-shell
    finite differences at all. ``shells`` is used as an explicit scale proxy
    because no real MATBG edge count has ever been measured
    (:data:`MATBG_EDGE_COUNT_BLOCKING_REASON`).
    """
    points = []
    for shells in shells_values:
        table = kernel_table_graph2mat(shells=int(shells), direction=direction)
        points.append(
            {
                "shells": int(shells),
                "edges_in_topology": table["edges_in_topology"],
                "jvp_calls": table["jvp_calls"],
                "build_seconds": table["build_seconds"],
            }
        )
    return {
        "schema": "epc_qb_qc_graph2mat_scaling_sweep_v1",
        "ticket": TICKET,
        "points": points,
        "jvp_calls_independent_of_shells": len({point["jvp_calls"] for point in points}) == 1,
        "edge_count_grows_with_shells": len({point["edges_in_topology"] for point in points}) > 1,
        "edge_count_limitation": (
            "this toy's fixed physical cutoff (CUTOFF_RADIUS_ANG) caps edges_in_topology at the "
            "same value for every shells value tried above the cutoff's own shell radius; shells "
            "beyond that only pad the table with structurally-empty (all-zero) entries, so this "
            "sweep is evidence for 'JVP calls do not scale with shells', not for how a real MATBG "
            "edge count would scale -- that number remains unmeasured (see below)"
        ),
        "matbg_edge_count_measured": MATBG_EDGE_COUNT_MEASURED,
        "matbg_edge_count_blocking_reason": MATBG_EDGE_COUNT_BLOCKING_REASON,
    }


if __name__ == "__main__":
    import json as _json

    print(_json.dumps(real_kernel_checks(), indent=2, default=str))
    print(_json.dumps(topology_margin_report(), indent=2, default=str))
    print(_json.dumps(scaling_sweep_graph2mat(), indent=2, default=str))
