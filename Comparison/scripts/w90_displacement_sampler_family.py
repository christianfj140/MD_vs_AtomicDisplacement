#!/usr/bin/env python3
"""Displacement sampler families for DATASET-DESIGN-W90-001-S2 (W90 graphene).

Geometry-only design/preview module: it builds displaced-configuration
metadata for the W90 graphene system (2 C atoms/cell, active-atom subsets of
size k=1 or k=2). It does not run SIESTA, does not train models, and does not
generate MD frames -- MD frame generation is step 6 (see ``md_family_contract``
below).

Cites and reuses conventions audited in
``docs/dataset_design_w90_001_s1_convention_manifest.md`` (S1):
  - Regime A geometry units (Angstrom, PBC lattice vectors always carried,
    native atom ordering) -- S1 section 2.
  - ``materials/graphene/RUN.fdf`` (2-atom primitive) and
    ``materials/graphene_5x5/RUN.fdf`` (50-atom supercell) as the large-
    supercell + active-subset reference pair -- S1 section 3.
  - ``materials/graphene/RUN.fdf`` uses ``AtomicCoordinatesFormat Fractional``,
    which ``shared/fdf_materialization.py`` deliberately rejects (see the
    docstring of ``canonical_ang_base_fdf`` in
    ``Comparison/scripts/run_epc_siesta_reference.py``). This module reuses
    the same fix already used there: read geometry via ``sisl`` instead of
    re-deriving fractional-coordinate parsing.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPHENE_PRIMITIVE_FDF = REPO_ROOT / "materials" / "graphene" / "RUN.fdf"
GRAPHENE_5X5_FDF = REPO_ROOT / "materials" / "graphene_5x5" / "RUN.fdf"
GRAPHENE_6X6_FDF = REPO_ROOT / "materials" / "graphene_6x6" / "RUN.fdf"

TRAIN_AMPLITUDES_ANG: tuple[float, ...] = (0.01, 0.03, 0.05, 0.08)
OOD_AMPLITUDE_ANG: float = 0.12
ALL_AMPLITUDES_ANG: tuple[float, ...] = TRAIN_AMPLITUDES_ANG + (OOD_AMPLITUDE_ANG,)
DIMENSIONALITIES: tuple[str, ...] = ("1D_in", "1D_z", "2D_in", "3D")
# Canonical sampler family ids: the generate_* functions below plus "MD"
# (MD_FAMILY_GENERATOR, a design-time contract only -- frames come from the
# MD pipeline, never fabricated here). Single source for pipeline_ui.py's
# dispatch and the UI's option list; add a sampler here first.
SAMPLER_IDS: tuple[str, ...] = (
    "axial_radial",
    "angular_shell",
    "angular_shell_collective",
    "local_pair_modes",
    "sobol_sparse",
    "random_cartesian",
    "MD",
)
NESTED_LEARNING_CURVE_SIZES: tuple[int, ...] = (16, 32, 64, 128, 256)
DEFAULT_BOND_CUTOFF_ANG: tuple[float, float] = (1.0, 1.8)

_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}
_SHIFT_RANGE = (-1, 0, 1)
_SHIFTS = np.array(
    [[a, b, c] for a in _SHIFT_RANGE for b in _SHIFT_RANGE for c in _SHIFT_RANGE],
    dtype=float,
)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    positions_ang: np.ndarray  # (N, 3)
    lattice_ang: np.ndarray  # (3, 3), rows are lattice vectors
    atomic_numbers: tuple[int, ...]

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)


def load_geometry(fdf_path: Path) -> Geometry:
    import sisl

    geometry = sisl.get_sile(str(fdf_path)).read_geometry()
    atoms = geometry.atoms
    atomic_numbers = tuple(int(z) for z in getattr(atoms, "Z", [atom.Z for atom in atoms]))
    return Geometry(
        positions_ang=np.array(geometry.xyz, dtype=float),
        lattice_ang=np.array(geometry.lattice.cell, dtype=float),
        atomic_numbers=atomic_numbers,
    )


def load_graphene_primitive() -> Geometry:
    return load_geometry(GRAPHENE_PRIMITIVE_FDF)


def load_graphene_5x5() -> Geometry:
    return load_geometry(GRAPHENE_5X5_FDF)


def load_graphene_6x6() -> Geometry:
    return load_geometry(GRAPHENE_6X6_FDF)


def min_image_vector(geometry: Geometry, i: int, j: int) -> tuple[np.ndarray, float]:
    """PBC-aware ``r_j - r_i`` under the minimum-image convention."""

    cart_shifts = _SHIFTS @ geometry.lattice_ang
    candidates = geometry.positions_ang[j] - geometry.positions_ang[i] + cart_shifts
    distances = np.linalg.norm(candidates, axis=1)
    best = int(np.argmin(distances))
    return candidates[best], float(distances[best])


def min_image_distance(geometry: Geometry, i: int, j: int) -> float:
    return min_image_vector(geometry, i, j)[1]


def nearest_neighbor(geometry: Geometry, i: int, *, exclude: Iterable[int] = ()) -> tuple[int, float]:
    excluded = set(exclude)
    best_j, best_d = None, math.inf
    for j in range(geometry.n_atoms):
        if j == i or j in excluded:
            continue
        distance = min_image_distance(geometry, i, j)
        if distance < best_d:
            best_j, best_d = j, distance
    if best_j is None:
        raise RuntimeError("nearest_neighbor found no candidate atom (structure has fewer than 2 atoms)")
    return best_j, best_d


def farthest_atom(geometry: Geometry, i: int) -> tuple[int, float]:
    best_j, best_d = None, -1.0
    for j in range(geometry.n_atoms):
        if j == i:
            continue
        distance = min_image_distance(geometry, i, j)
        if distance > best_d:
            best_j, best_d = j, distance
    if best_j is None:
        raise RuntimeError("farthest_atom found no candidate atom (structure has fewer than 2 atoms)")
    return best_j, best_d


def pairwise_shell_rank(geometry: Geometry, i: int, j: int) -> int:
    """1-based neighbour-shell rank of atom ``j`` around atom ``i`` (1 = nearest shell)."""

    distances = sorted({round(min_image_distance(geometry, i, m), 6) for m in range(geometry.n_atoms) if m != i})
    target = round(min_image_distance(geometry, i, j), 6)
    return distances.index(target) + 1


# --------------------------------------------------------------------------
# Active-atom selection (k independent of N)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveAtomSelection:
    indices: tuple[int, ...]
    k: int
    connected: bool
    shell_span: int  # 0 for k=1; pair shell for k=2; max bond-hop depth otherwise
    pair_distance_ang: float | None


def select_active_atoms(
    geometry: Geometry,
    k: int,
    *,
    center_index: int = 0,
    pair_mode: str = "bonded",
    bond_cutoff_ang: tuple[float, float] = DEFAULT_BOND_CUTOFF_ANG,
) -> ActiveAtomSelection:
    """Select k active atoms without assuming N==2 (works for a 2-atom cell or an N>2 supercell)."""

    n_atoms_total = geometry.n_atoms
    if not 1 <= k <= n_atoms_total:
        raise ValueError(f"k must be between 1 and {n_atoms_total}, got {k}")
    if not 0 <= center_index < n_atoms_total:
        raise ValueError(f"center_index must be between 0 and {n_atoms_total - 1}, got {center_index}")
    if k == 1:
        return ActiveAtomSelection((center_index,), 1, True, 0, None)
    if k == 2:
        if pair_mode == "bonded":
            partner, distance = nearest_neighbor(geometry, center_index)
        elif pair_mode == "arbitrary":
            partner, distance = farthest_atom(geometry, center_index)
        else:
            raise ValueError(f"unknown pair_mode {pair_mode!r}; expected 'bonded' or 'arbitrary'")
        rank = pairwise_shell_rank(geometry, center_index, partner)
        connected = bond_cutoff_ang[0] <= distance <= bond_cutoff_ang[1]
        return ActiveAtomSelection((center_index, partner), 2, connected, rank, distance)
    if k == n_atoms_total:
        # "Move every atom" (e.g. a 6x6 supercell's independent-per-atom sampler
        # families -- sobol_sparse/random_cartesian/latin_hypercube all give each
        # active atom its own displacement already, so this is a real generalization,
        # not a rigid shift). connected/shell_span are pairwise-only concepts (k=2);
        # trivially True/0 here since there's no pair to (dis)connect.
        return ActiveAtomSelection(tuple(range(n_atoms_total)), n_atoms_total, True, 0, None)
    if pair_mode != "bonded":
        raise ValueError("k > 2 requires pair_mode='bonded'")

    # Deterministic breadth-first traversal of the periodic bond graph. Every
    # prefix is therefore connected and nested in every larger-k selection.
    lower, upper = bond_cutoff_ang
    order, depth, queue = [center_index], {center_index: 0}, [center_index]
    while queue:
        atom = queue.pop(0)
        neighbours = sorted(
            (min_image_distance(geometry, atom, other), other)
            for other in range(n_atoms_total)
            if other not in depth and lower <= min_image_distance(geometry, atom, other) <= upper
        )
        for _distance, other in neighbours:
            depth[other] = depth[atom] + 1
            order.append(other)
            queue.append(other)
    if len(order) != n_atoms_total:
        raise RuntimeError(f"bond graph is disconnected: reached {len(order)}/{n_atoms_total} atoms")
    selected = tuple(order[:k])
    return ActiveAtomSelection(selected, k, True, max(depth[index] for index in selected), None)


# --------------------------------------------------------------------------
# Dimensionality axis sets
# --------------------------------------------------------------------------


def dimensionality_axes(dim: str) -> tuple[np.ndarray, ...]:
    if dim == "1D_in":
        return (_AXES["x"],)
    if dim == "1D_z":
        return (_AXES["z"],)
    if dim == "2D_in":
        return (_AXES["x"], _AXES["y"])
    if dim == "3D":
        return (_AXES["x"], _AXES["y"], _AXES["z"])
    raise ValueError(f"unknown dimensionality {dim!r}; expected one of {DIMENSIONALITIES}")


def axial_directions(dim: str) -> list[np.ndarray]:
    return [sign * axis for axis in dimensionality_axes(dim) for sign in (1.0, -1.0)]


def envelope_half_width(amplitude_ang: float, n_axes: int) -> float:
    """Per-component box half-width whose worst-case L2 corner norm is ``amplitude_ang``.

    ``axial_radial``/``angular_shell`` place every displacement at L2 norm exactly
    ``amplitude_ang`` (a fixed-radius shell). The box-sampled families below
    (``sobol_sparse``, ``random_cartesian``, ``latin_hypercube``) instead drew each
    axis independently in ``[-amplitude_ang, amplitude_ang]``, so their worst-case
    corner had norm ``amplitude_ang * sqrt(n_axes)`` -- silently exceeding the same
    ``amplitude_ang`` = R_train_max envelope every other family respects. Scaling the
    half-width by ``1/sqrt(n_axes)`` makes ``amplitude_ang`` a shared per-atom max-norm
    bound (R_train_max) across every family instead of a per-family-specific meaning.
    """

    return amplitude_ang / math.sqrt(n_axes) if n_axes else amplitude_ang


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """Quasi-uniform points on the unit sphere (Fibonacci spiral)."""

    indices = np.arange(n_points)
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    z = 1.0 - 2.0 * (indices + 0.5) / n_points
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    golden_angle = 2.0 * np.pi * (1.0 - 1.0 / golden_ratio)
    phi = golden_angle * indices
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------
# Configuration + metadata
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Configuration:
    family: str
    displacements_ang: dict[int, tuple[float, float, float]]
    metadata: dict[str, Any]


def build_metadata(
    geometry: Geometry,
    active: ActiveAtomSelection,
    displacements_ang: dict[int, tuple[float, float, float]],
    *,
    family: str,
    dim: str,
    amplitude_ang: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_total = geometry.n_atoms
    k = active.k
    vectors = np.array([displacements_ang[index] for index in active.indices])
    norms = np.linalg.norm(vectors, axis=1)
    sum_sq = float((norms**2).sum())
    sum_sq4 = float((norms**4).sum())
    participation_ratio = (sum_sq**2) / (n_total * sum_sq4) if sum_sq4 > 0 else 0.0
    rms = math.sqrt(sum_sq / k) if k else 0.0
    max_disp = float(norms.max()) if k else 0.0

    positions = geometry.positions_ang[list(active.indices)]
    centroid = positions.mean(axis=0)
    radius_of_gyration = math.sqrt(float(np.mean(np.sum((positions - centroid) ** 2, axis=1)))) if k else 0.0

    correlation = None
    if k == 2 and norms[0] > 1e-12 and norms[1] > 1e-12:
        correlation = float(np.dot(vectors[0], vectors[1]) / (norms[0] * norms[1]))

    metadata = {
        "family": family,
        "dimensionality": dim,
        "amplitude_ang": amplitude_ang,
        "k": k,
        "n_atoms_total": n_total,
        "k_over_n": k / n_total,
        "participation_ratio": participation_ratio,
        "rms_displacement_ang": rms,
        "max_displacement_ang": max_disp,
        "neighbor_shell_span": active.shell_span,
        "radius_of_gyration_ang": radius_of_gyration,
        "connected": active.connected,
        "correlation": correlation,
        "active_atom_indices": list(active.indices),
    }
    if extra:
        metadata.update(extra)
    return metadata


def nested_learning_curve_prefixes(
    configs: Sequence[Configuration],
    sizes: Sequence[int] = NESTED_LEARNING_CURVE_SIZES,
) -> dict[int, list[Configuration]]:
    """Prefixes of one max-N generation call; a prefix is a subset by construction.

    For ``sobol_sparse`` the prefix is additionally a valid low-discrepancy
    Sobol net (scipy guarantees this for a single continuous generation call
    starting at index 0), which is the stronger "nested sequence" property
    requested for Sobol/radial-shell families.
    """

    return {n: list(configs[:n]) for n in sizes if n <= len(configs)}


# --------------------------------------------------------------------------
# Family 1: axial_radial
# --------------------------------------------------------------------------


def generate_axial_radial(
    geometry: Geometry,
    k: int,
    dim: str,
    *,
    radii_ang: Sequence[float] = ALL_AMPLITUDES_ANG,
    center_index: int = 0,
    pair_mode: str = "bonded",
) -> list[Configuration]:
    """Displace the active atom(s) along canonical +/-axis directions at explicit radial shells."""

    active = select_active_atoms(geometry, k, center_index=center_index, pair_mode=pair_mode)
    configs: list[Configuration] = []
    for radius in radii_ang:
        for direction in axial_directions(dim):
            vector = tuple((radius * direction).tolist())
            displacements = {index: vector for index in active.indices}
            metadata = build_metadata(
                geometry,
                active,
                displacements,
                family="axial_radial",
                dim=dim,
                amplitude_ang=radius,
                extra={"direction": direction.tolist()},
            )
            configs.append(Configuration("axial_radial", displacements, metadata))
    return configs


# --------------------------------------------------------------------------
# Family 2: angular_shell
# --------------------------------------------------------------------------


def generate_angular_shell(
    geometry: Geometry,
    k: int,
    dim: str,
    radius_ang: float,
    *,
    n_points: int = 24,
    center_index: int = 0,
    pair_mode: str = "bonded",
) -> list[Configuration]:
    """Fixed-radius shell with uniform (2D) or quasi-uniform Fibonacci-sphere (3D) angular coverage."""

    if dim not in ("2D_in", "3D"):
        raise ValueError(
            f"angular_shell only supports 2D_in (uniform angles) and 3D (quasi-uniform sphere) dimensionality, got {dim!r}"
        )
    active = select_active_atoms(geometry, k, center_index=center_index, pair_mode=pair_mode)
    if dim == "2D_in":
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    else:
        directions = _fibonacci_sphere(n_points)

    configs: list[Configuration] = []
    for angle_index, direction in enumerate(directions):
        vector = tuple((radius_ang * direction).tolist())
        displacements = {index: vector for index in active.indices}
        metadata = build_metadata(
            geometry,
            active,
            displacements,
            family="angular_shell",
            dim=dim,
            amplitude_ang=radius_ang,
            extra={"direction": direction.tolist(), "angle_index": angle_index},
        )
        configs.append(Configuration("angular_shell", displacements, metadata))
    return configs


def generate_angular_shell_collective(
    geometry: Geometry,
    radius_ang: float,
    n_structures: int,
    *,
    seed: int = 0,
) -> list[Configuration]:
    """Move every atom by a fixed radius in antipodal pairs, with zero net shift."""

    if geometry.n_atoms % 2:
        raise ValueError("angular_shell_collective requires an even number of atoms")
    active = select_active_atoms(geometry, geometry.n_atoms)
    rng = np.random.default_rng(seed)
    configs: list[Configuration] = []
    for _ in range(n_structures):
        directions = rng.normal(size=(geometry.n_atoms // 2, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.concatenate((directions, -directions))
        directions = directions[rng.permutation(geometry.n_atoms)]
        displacements = {index: tuple(radius_ang * directions[index]) for index in active.indices}
        metadata = build_metadata(
            geometry,
            active,
            displacements,
            family="angular_shell_collective",
            dim="3D",
            amplitude_ang=radius_ang,
            extra={"seed": seed, "antipodal_pairs": geometry.n_atoms // 2},
        )
        configs.append(Configuration("angular_shell_collective", displacements, metadata))
    return configs


# --------------------------------------------------------------------------
# Family 3: local_pair_modes (k=2 only)
# --------------------------------------------------------------------------


def bond_frame(geometry: Geometry, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Orthonormal (longitudinal, transverse in-plane, transverse out-of-plane) frame for bond i-j."""

    vector, distance = min_image_vector(geometry, i, j)
    e_l = vector / distance
    z_hat = _AXES["z"]
    if abs(float(np.dot(e_l, z_hat))) > 0.999:
        e_t1 = _AXES["x"] - np.dot(_AXES["x"], e_l) * e_l
        e_t1 = e_t1 / np.linalg.norm(e_t1)
        e_t2 = np.cross(e_l, e_t1)
    else:
        e_t2 = z_hat - np.dot(z_hat, e_l) * e_l
        norm = np.linalg.norm(e_t2)
        e_t2 = e_t2 / norm if norm > 1e-9 else z_hat
        e_t1 = np.cross(e_t2, e_l)
        e_t1 = e_t1 / np.linalg.norm(e_t1)
    return e_l, e_t1, e_t2, distance


def generate_local_pair_modes(
    geometry: Geometry,
    dim: str,
    *,
    amplitudes_ang: Sequence[float] = ALL_AMPLITUDES_ANG,
    center_index: int = 0,
) -> list[Configuration]:
    """5 pair modes for k=2: longitudinal/transverse_ip/transverse_opp are the bond-frame
    antiparallel (optical-like) modes; parallel/antiparallel are lab-axis in-phase/
    out-of-phase modes, each gated by ``dim``."""

    active = select_active_atoms(geometry, 2, center_index=center_index, pair_mode="bonded")
    i, j = active.indices
    e_l, e_t1, e_t2, _bond_len = bond_frame(geometry, i, j)
    axes = dimensionality_axes(dim)

    in_plane_allowed = dim in ("1D_in", "2D_in", "3D")
    two_in_plane_allowed = dim in ("2D_in", "3D")
    out_of_plane_allowed = dim in ("1D_z", "3D")

    configs: list[Configuration] = []

    def emit(mode: str, direction: np.ndarray, partner_sign: float, amplitude: float) -> None:
        u = amplitude * direction
        displacements = {i: tuple(u.tolist()), j: tuple((partner_sign * u).tolist())}
        metadata = build_metadata(
            geometry,
            active,
            displacements,
            family="local_pair_modes",
            dim=dim,
            amplitude_ang=amplitude,
            extra={"mode": mode},
        )
        configs.append(Configuration("local_pair_modes", displacements, metadata))

    for amplitude in amplitudes_ang:
        if in_plane_allowed:
            emit("longitudinal", e_l, -1.0, amplitude)
        if two_in_plane_allowed:
            emit("transverse_ip", e_t1, -1.0, amplitude)
        if out_of_plane_allowed:
            emit("transverse_opp", e_t2, -1.0, amplitude)
        for axis in axes:
            emit("parallel", axis, +1.0, amplitude)
            emit("antiparallel", axis, -1.0, amplitude)
    return configs


# --------------------------------------------------------------------------
# Family 4: sobol_sparse (scrambled Sobol, nested by construction)
# --------------------------------------------------------------------------


def generate_sobol_sparse(
    geometry: Geometry,
    k: int,
    dim: str,
    amplitude_ang: float,
    *,
    max_n: int = 256,
    seed: int = 0,
    center_index: int = 0,
    pair_mode: str = "bonded",
) -> list[Configuration]:
    """Scrambled-Sobol displacements inside a per-axis +/-amplitude box.

    Generated as one continuous ``sampler.random(max_n)`` call so that any
    power-of-two prefix (16, 32, 64, 128, 256, ...) is itself a valid Sobol
    net -- the nested-sequence property required by the task spec.
    """

    from scipy.stats import qmc

    active = select_active_atoms(geometry, k, center_index=center_index, pair_mode=pair_mode)
    axes = dimensionality_axes(dim)
    dims = k * len(axes)
    half_width = envelope_half_width(amplitude_ang, len(axes))
    sampler = qmc.Sobol(d=dims, scramble=True, seed=seed)
    unit_points = sampler.random(max_n)

    configs: list[Configuration] = []
    for row in unit_points:
        coefficients = row.reshape(k, len(axes))
        displacements: dict[int, tuple[float, float, float]] = {}
        for slot, atom_index in enumerate(active.indices):
            vector = np.zeros(3)
            for coeff, axis in zip(coefficients[slot], axes):
                vector += half_width * (2.0 * coeff - 1.0) * axis
            displacements[atom_index] = tuple(vector.tolist())
        metadata = build_metadata(
            geometry,
            active,
            displacements,
            family="sobol_sparse",
            dim=dim,
            amplitude_ang=amplitude_ang,
        )
        configs.append(Configuration("sobol_sparse", displacements, metadata))
    return configs


# --------------------------------------------------------------------------
# Family 5: random_cartesian (reuses the bounded-gaussian/uniform pattern)
# --------------------------------------------------------------------------


def generate_random_cartesian(
    geometry: Geometry,
    k: int,
    dim: str,
    amplitude_ang: float,
    n_structures: int,
    *,
    seed: int = 0,
    distribution: str = "uniform",
    center_index: int = 0,
    pair_mode: str = "bonded",
) -> list[Configuration]:
    """Per-axis bounded gaussian/uniform noise on the k active atoms.

    Same bounded-gaussian / uniform-box pattern as
    ``AtomDisplacement/scripts/generate_random_cartesian_dataset.py::bounded_gaussian_vector``,
    generalized to k active atoms and dimensionality-axis masking (that
    helper assumes displacing every atom in full 3D).
    """

    import random as _random

    if distribution not in ("uniform", "gaussian"):
        raise ValueError(f"distribution must be 'uniform' or 'gaussian', got {distribution!r}")

    active = select_active_atoms(geometry, k, center_index=center_index, pair_mode=pair_mode)
    axes = dimensionality_axes(dim)
    half_width = envelope_half_width(amplitude_ang, len(axes))
    rng = _random.Random(seed)

    configs: list[Configuration] = []
    for _sample_index in range(n_structures):
        displacements: dict[int, tuple[float, float, float]] = {}
        for atom_index in active.indices:
            vector = np.zeros(3)
            for axis in axes:
                if distribution == "gaussian":
                    component = half_width
                    for _attempt in range(1000):
                        candidate = rng.gauss(0.0, half_width / 3.0)
                        if abs(candidate) <= half_width:
                            component = candidate
                            break
                else:
                    component = rng.uniform(-half_width, half_width)
                vector += component * axis
            displacements[atom_index] = tuple(vector.tolist())
        metadata = build_metadata(
            geometry,
            active,
            displacements,
            family="random_cartesian",
            dim=dim,
            amplitude_ang=amplitude_ang,
            extra={"distribution": distribution},
        )
        configs.append(Configuration("random_cartesian", displacements, metadata))
    return configs


# --------------------------------------------------------------------------
# Family 7: latin_hypercube -- frozen-test-only sampler (DATASET-DESIGN-W90-001-S5)
#
# Distinct QMC construction from ``generate_sobol_sparse``: Latin Hypercube
# Sampling stratifies each 1-D marginal into ``n_structures`` equal bins and
# permutes independently per dimension (no digital-net structure), and here it
# additionally samples the displacement amplitude jointly with direction
# (continuous draw inside ``amplitude_range_ang``) instead of taking amplitude
# as a fixed per-call radius the way every training-time family above does.
# Used only to build the S5 ``common_displacement`` frozen test -- it must
# never appear in ``run_w90_displacement_siesta_pilot.py`` (S3) training scope.
# --------------------------------------------------------------------------

COMMON_DISPLACEMENT_AMPLITUDE_RANGE_ANG: tuple[float, float] = (0.03, 0.08)


def generate_latin_hypercube(
    geometry: Geometry,
    k: int,
    dim: str,
    *,
    n_structures: int,
    amplitude_range_ang: tuple[float, float] = COMMON_DISPLACEMENT_AMPLITUDE_RANGE_ANG,
    seed: int = 0,
    center_index: int = 0,
    pair_mode: str = "bonded",
) -> list[Configuration]:
    """LHS over direction *and* amplitude jointly; amplitudes mix across ``amplitude_range_ang``."""

    from scipy.stats import qmc

    active = select_active_atoms(geometry, k, center_index=center_index, pair_mode=pair_mode)
    axes = dimensionality_axes(dim)
    dims = k * len(axes) + 1  # +1 for the amplitude marginal
    sampler = qmc.LatinHypercube(d=dims, seed=seed)
    unit_points = sampler.random(n_structures)

    configs: list[Configuration] = []
    for row in unit_points:
        amplitude = amplitude_range_ang[0] + row[-1] * (amplitude_range_ang[1] - amplitude_range_ang[0])
        half_width = envelope_half_width(amplitude, len(axes))
        coefficients = row[:-1].reshape(k, len(axes))
        displacements: dict[int, tuple[float, float, float]] = {}
        for slot, atom_index in enumerate(active.indices):
            vector = np.zeros(3)
            for coeff, axis in zip(coefficients[slot], axes):
                vector += half_width * (2.0 * coeff - 1.0) * axis
            displacements[atom_index] = tuple(vector.tolist())
        metadata = build_metadata(
            geometry,
            active,
            displacements,
            family="latin_hypercube",
            dim=dim,
            amplitude_ang=amplitude,
        )
        configs.append(Configuration("latin_hypercube", displacements, metadata))
    return configs


# --------------------------------------------------------------------------
# Family 6: MD -- design-time contract only, no frame generation (step 6)
# --------------------------------------------------------------------------

MD_FAMILY_GENERATOR = "MD/scripts/generate_md_dataset.py"


def md_family_contract() -> dict[str, Any]:
    """Design-time contract for the MD family.

    DATASET-DESIGN-W90-001-S2 must not generate MD frames (explicit task
    constraint; that is step 6). This records which existing generator will
    own frame production, per S1 section 4, without running it.
    """

    return {
        "family": "MD",
        "status": "not_generated_by_this_module",
        "generator": MD_FAMILY_GENERATOR,
        "reason": "MD frame generation is out of scope for DATASET-DESIGN-W90-001-S2 (reserved for step 6).",
        "dimensionalities": DIMENSIONALITIES,
        "amplitudes_ang": ALL_AMPLITUDES_ANG,
    }


# --------------------------------------------------------------------------
# 5x5 supercell compatibility preview (geometry only, no SIESTA)
# --------------------------------------------------------------------------


def compatibility_preview(
    geometry: Geometry,
    k: int,
    *,
    center_index: int = 0,
    pair_mode: str = "bonded",
) -> dict[str, Any]:
    """Select an active k-atom subset from a supercell (N>2) without assuming N==2."""

    active = select_active_atoms(geometry, k, center_index=center_index, pair_mode=pair_mode)
    return {
        "n_atoms_total": geometry.n_atoms,
        "k": active.k,
        "active_atom_indices": list(active.indices),
        "connected": active.connected,
        "neighbor_shell_span": active.shell_span,
        "pair_distance_ang": active.pair_distance_ang,
    }
