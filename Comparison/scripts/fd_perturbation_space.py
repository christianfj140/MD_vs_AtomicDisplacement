#!/usr/bin/env python3
"""C08 / E-F_001-S10: the finite-difference perturbation space.

Everything a central-difference derivative needs *before* any matrix exists:
which displacement patterns are allowed, how they are normalised, which
amplitudes are swept, and when a displacement must be refused because it moves
an atom across a neighbour cutoff.

A perturbation is one :class:`Direction`

    v in R^(N x 3),   ||v||_F = 1

and one amplitude ``delta`` in Ang, so the stencil geometries are ``R +- delta v``.
Three families are defined, and the roadmap's graphene-Gamma experiment uses
all three:

``one_hot``
    ``v = e_(I,alpha)``. Already unit-Frobenius, so ``delta v`` is *exactly* the
    single-coordinate displacement the stencil builder has always written --
    the legacy path is this path, not a parallel one.
``collective``
    any non-trivial pattern; :func:`random_collective` builds a deterministic
    seeded one for linearity / arbitrary-``v`` tests (GO-3).
``uniform_translation``
    every atom moved alike. Unit-Frobenius means each atom moves
    ``delta/sqrt(N)``; H must be invariant, which is the acoustic control.

Normalisation is ``||v||_F = 1`` and nothing else (:data:`NORMALIZATION`):
it makes ``delta`` the total displacement amplitude for every family at once,
and it is the only convention under which the one-hot case degenerates to the
historical ``+-delta`` on one coordinate.

The amplitudes are a *sweep*, never a preference. :func:`delta_sweep` builds the
geometric triple around an operating centre and :func:`select_delta_from_plateau`
is the only function here allowed to name a winner -- and it needs measured
values to do it. A single ``delta`` cannot produce a plateau, so
:func:`delta_sweep_status` marks such a build ``insufficient_for_plateau``.

Topology is the hard gate. Graph2Mat derivatives hold ``edge_index``/``shifts``
fixed and SIESTA's sparsity follows ``d_ij <= rc_i + rc_j``; a displacement that
takes a pair across that radius is a discontinuity of the model, not a
derivative. :func:`topology_margin` enumerates every pair that could cross under
``+-delta v`` (including periodic images) and reports the margin;
:func:`assert_topology_preserved` refuses the perturbation if any pair does.

Units: positions, cutoffs, deltas and margins are all Ang.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from artifact_signature import input_signature_sha256  # noqa: E402

PERTURBATION_SCHEMA = "fd_perturbation_direction_v1"

# The one normalisation of this repository. Read the module docstring before
# adding a second one: every delta, margin and tolerance below assumes it.
NORMALIZATION = "unit_frobenius"

KIND_ONE_HOT = "one_hot"
KIND_COLLECTIVE = "collective"
KIND_UNIFORM_TRANSLATION = "uniform_translation"
VALID_KINDS = (KIND_ONE_HOT, KIND_COLLECTIVE, KIND_UNIFORM_TRANSLATION)

AXES = {"x": 0, "y": 1, "z": 2}
AXIS_NAMES = ("x", "y", "z")

VALID_METHODS = {"central", "forward", "backward"}

# Operating regime of the existing derivative campaigns (docs/workflows.md uses
# 0.005-0.01 Ang; the small-delta campaign went down to 5e-4). The centre is a
# starting point for the sweep, NOT a chosen delta.
DEFAULT_DELTA_CENTER_ANG = 0.01
DEFAULT_DELTA_RATIO = 2.0
MIN_PLATEAU_DELTA_COUNT = 3
DEFAULT_DELTA_SWEEP_ANG = (0.005, 0.01, 0.02)

# Below this the direction is numerically indistinguishable from zero.
MIN_DIRECTION_NORM = 1e-12


class FdPerturbationError(ValueError):
    """A perturbation was requested that the FD space does not admit."""


@dataclass(frozen=True)
class Direction:
    """A normalised displacement pattern ``v`` of shape ``(N, 3)``.

    Build through :func:`one_hot`, :func:`uniform_translation`,
    :func:`collective` or :func:`random_collective`; the constructor does not
    normalise, so that a hand-made instance cannot silently carry a different
    convention than the one it declares.
    """

    name: str
    kind: str
    vectors: np.ndarray
    atom_index_zero_based: int | None = None
    axis: str | None = None
    normalization: str = NORMALIZATION

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise FdPerturbationError(f"Unsupported perturbation kind: {self.kind!r}.")
        if self.normalization != NORMALIZATION:
            raise FdPerturbationError(f"Only {NORMALIZATION!r} directions are supported.")
        if not str(self.name).strip():
            raise FdPerturbationError("A direction needs a name; it becomes the sample id.")
        vectors = np.asarray(self.vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise FdPerturbationError(f"Direction vectors must have shape (N, 3); got {vectors.shape}.")
        norm = float(np.linalg.norm(vectors))
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-9:
            raise FdPerturbationError(
                f"Direction {self.name!r} is not unit-Frobenius (||v||_F = {norm!r}); use normalize()."
            )
        object.__setattr__(self, "vectors", vectors)

    @property
    def atom_count(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def axis_index(self) -> int | None:
        return AXES[self.axis] if self.axis in AXES else None

    @property
    def max_atom_norm(self) -> float:
        """Largest per-atom displacement per unit ``delta``."""
        return float(np.max(np.linalg.norm(self.vectors, axis=1))) if self.atom_count else 0.0

    @property
    def direction_hash(self) -> str:
        return input_signature_sha256(
            {
                "schema": PERTURBATION_SCHEMA,
                "kind": self.kind,
                "normalization": self.normalization,
                "vectors": self.vectors.tolist(),
            }
        )

    def to_metadata(self) -> dict[str, Any]:
        """The fields that travel into sample metadata and artifact signatures."""
        payload: dict[str, Any] = {
            "direction_schema": PERTURBATION_SCHEMA,
            "direction_name": self.name,
            "direction_kind": self.kind,
            "direction_normalization": self.normalization,
            "direction_hash": self.direction_hash,
            "direction_atom_count": self.atom_count,
            "direction_max_atom_norm": self.max_atom_norm,
        }
        if self.kind == KIND_ONE_HOT:
            payload["atom_index_zero_based"] = self.atom_index_zero_based
            payload["axis"] = self.axis
            payload["axis_index"] = self.axis_index
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation, vectors included."""
        return {**self.to_metadata(), "vectors": self.vectors.tolist()}


def normalize(vectors: Any, *, name: str = "direction") -> np.ndarray:
    """Return ``v / ||v||_F`` as float64 ``(N, 3)``."""
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise FdPerturbationError(f"{name}: displacement pattern must have shape (N, 3); got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise FdPerturbationError(f"{name}: displacement pattern has non-finite entries.")
    norm = float(np.linalg.norm(array))
    if norm < MIN_DIRECTION_NORM:
        raise FdPerturbationError(f"{name}: displacement pattern is zero (||v||_F = {norm:g}).")
    return array / norm


def _axis_name(axis: str | int) -> str:
    """``0/'x'`` -> ``'x'``; anything else is left unresolved for the caller to reject.

    An out-of-range integer is *not* wrapped around: ``-1`` is a mistake, not the
    z axis.
    """
    if isinstance(axis, (int, np.integer)):
        return AXIS_NAMES[int(axis)] if 0 <= int(axis) < len(AXIS_NAMES) else ""
    return str(axis).strip().lower()


def one_hot(atom_count: int, atom_index_zero_based: int, axis: str | int) -> Direction:
    """``v = e_(I,alpha)``: the historical single-coordinate stencil."""
    axis_name = _axis_name(axis)
    if axis_name not in AXES:
        raise FdPerturbationError(f"Unsupported axis: {axis!r}. Use x, y or z.")
    if atom_count <= 0:
        raise FdPerturbationError("atom_count must be positive.")
    if not 0 <= int(atom_index_zero_based) < atom_count:
        raise FdPerturbationError(
            f"atom index {atom_index_zero_based} is outside a structure with {atom_count} atoms."
        )
    vectors = np.zeros((atom_count, 3), dtype=np.float64)
    vectors[int(atom_index_zero_based), AXES[axis_name]] = 1.0
    return Direction(
        name=f"atom{int(atom_index_zero_based):04d}_{axis_name}",
        kind=KIND_ONE_HOT,
        vectors=vectors,
        atom_index_zero_based=int(atom_index_zero_based),
        axis=axis_name,
    )


def uniform_translation(atom_count: int, axis_or_vector: str | int | Sequence[float] = "x") -> Direction:
    """Every atom moved by the same vector: the acoustic / invariance control."""
    if atom_count <= 0:
        raise FdPerturbationError("atom_count must be positive.")
    if isinstance(axis_or_vector, (str, int, np.integer)):
        axis_name = _axis_name(axis_or_vector)
        if axis_name not in AXES:
            raise FdPerturbationError(f"Unsupported translation axis: {axis_or_vector!r}.")
        step = np.zeros(3, dtype=np.float64)
        step[AXES[axis_name]] = 1.0
        label = axis_name
    else:
        step = np.asarray(axis_or_vector, dtype=np.float64).reshape(-1)
        if step.shape != (3,):
            raise FdPerturbationError("A translation vector must have three components.")
        label = "_".join(f"{value:g}" for value in step)
    return Direction(
        name=f"translation_{label}",
        kind=KIND_UNIFORM_TRANSLATION,
        vectors=normalize(np.tile(step, (atom_count, 1)), name="uniform_translation"),
    )


def collective(vectors: Any, *, name: str) -> Direction:
    """An arbitrary non-trivial pattern, normalised on the way in."""
    return Direction(name=name, kind=KIND_COLLECTIVE, vectors=normalize(vectors, name=name))


def random_collective(atom_count: int, *, seed: int = 0) -> Direction:
    """A deterministic pseudo-random collective pattern (arbitrary-``v`` tests)."""
    if atom_count <= 0:
        raise FdPerturbationError("atom_count must be positive.")
    generator = np.random.default_rng(int(seed))
    return collective(generator.standard_normal((int(atom_count), 3)), name=f"random_seed{int(seed)}")


def default_direction_set(atom_count: int, *, seed: int = 0) -> list[Direction]:
    """The minimum space S10 requires: one Cartesian, one collective, one translation."""
    return [
        one_hot(atom_count, 0, "x"),
        random_collective(atom_count, seed=seed),
        uniform_translation(atom_count, "x"),
    ]


# --------------------------------------------------------------------------- #
# Synthetic non-phonon resource-benchmark directions (E-F_001-S37 / D06)
#
# These give a JVP/contraction cost model the right *shape* of collective mode
# (interlayer-normal, interlayer in-plane, alternating-atom) without claiming
# to be a diagonalised dynamical-matrix eigenvector: no mass weighting, no
# frequency, no phonon provider. Callers must keep them tagged
# ``synthetic_test_displacement`` and never ``phonon_eigenmode`` (roadmap
# Fase 10b / GO-8).
# --------------------------------------------------------------------------- #


def breathing_like(positions_ang: Any, *, name: str = "breathing_like") -> Direction:
    """Two halves (split by median z) move along +/-z: an interlayer-normal probe."""
    positions = np.asarray(positions_ang, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise FdPerturbationError(f"positions must have shape (N, 3); got {positions.shape}")
    vectors = np.zeros_like(positions)
    vectors[:, 2] = np.where(positions[:, 2] >= np.median(positions[:, 2]), 1.0, -1.0)
    return collective(vectors, name=name)


def shear_like(positions_ang: Any, axis: str | int = "x", *, name: str | None = None) -> Direction:
    """Two halves (split by median z) slide antiparallel along an in-plane axis."""
    positions = np.asarray(positions_ang, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise FdPerturbationError(f"positions must have shape (N, 3); got {positions.shape}")
    axis_name = _axis_name(axis)
    if axis_name not in ("x", "y"):
        raise FdPerturbationError(f"shear_like acts in-plane; unsupported axis {axis!r}.")
    vectors = np.zeros_like(positions)
    vectors[:, AXES[axis_name]] = np.where(
        positions[:, 2] >= np.median(positions[:, 2]), 1.0, -1.0
    )
    return collective(vectors, name=name or f"shear_like_{axis_name}")


def optical_like(atom_count: int, *, name: str = "optical_like", seed: int = 0) -> Direction:
    """Even/odd atom index move antiparallel along one random axis: an alternating-atom probe."""
    if atom_count <= 0:
        raise FdPerturbationError("atom_count must be positive.")
    generator = np.random.default_rng(int(seed))
    axis_vector = generator.standard_normal(3)
    sign = np.where(np.arange(int(atom_count)) % 2 == 0, 1.0, -1.0)
    vectors = sign[:, None] * axis_vector[None, :]
    return collective(vectors, name=name)


def synthetic_benchmark_direction_set(positions_ang: Any, *, seed: int = 0) -> list[Direction]:
    """The four families E-F_001-S37 requires: breathing/shear/optical/random."""
    positions = np.asarray(positions_ang, dtype=np.float64)
    atom_count = int(positions.shape[0])
    return [
        breathing_like(positions),
        shear_like(positions, "x"),
        optical_like(atom_count, seed=seed),
        random_collective(atom_count, seed=seed),
    ]


def displace(
    positions_ang: Sequence[Sequence[float]],
    direction: Direction,
    signed_delta_ang: float,
) -> list[list[float]]:
    """Return ``R + signed_delta * v`` in Ang.

    Components where ``v`` is exactly zero are copied, not summed, so a one-hot
    direction reproduces the single-coordinate stencil bit for bit.
    """
    positions = [list(map(float, position)) for position in positions_ang]
    if len(positions) != direction.atom_count:
        raise FdPerturbationError(
            f"Direction {direction.name!r} spans {direction.atom_count} atoms but the structure has {len(positions)}."
        )
    delta = float(signed_delta_ang)
    for atom_index, row in enumerate(direction.vectors):
        for component_index, component in enumerate(row):
            if component != 0.0:
                positions[atom_index][component_index] += delta * float(component)
    return positions


def signs_for_method(method: str) -> tuple[int, ...]:
    method = str(method or "").strip().lower()
    if method == "central":
        return (1, -1)
    if method == "forward":
        return (1,)
    if method == "backward":
        return (-1,)
    raise FdPerturbationError(f"Unsupported finite difference method: {method!r}.")


# --------------------------------------------------------------------------- #
# Amplitudes
# --------------------------------------------------------------------------- #


def delta_sweep(
    center_ang: float = DEFAULT_DELTA_CENTER_ANG,
    *,
    ratio: float = DEFAULT_DELTA_RATIO,
    count: int = MIN_PLATEAU_DELTA_COUNT,
) -> tuple[float, ...]:
    """A geometric sweep of ``count`` amplitudes centred on ``center_ang``."""
    if center_ang <= 0:
        raise FdPerturbationError("delta centre must be positive.")
    if ratio <= 1:
        raise FdPerturbationError("delta sweep ratio must be > 1.")
    if count < MIN_PLATEAU_DELTA_COUNT or count % 2 == 0:
        raise FdPerturbationError(f"A delta sweep needs an odd count >= {MIN_PLATEAU_DELTA_COUNT}.")
    half = count // 2
    return tuple(float(center_ang) * float(ratio) ** exponent for exponent in range(-half, half + 1))


def delta_sweep_status(delta_ang_values: Sequence[float]) -> dict[str, Any]:
    """Whether this set of amplitudes can exhibit a plateau at all."""
    deltas = sorted({float(value) for value in delta_ang_values})
    if not deltas or deltas[0] <= 0:
        raise FdPerturbationError("delta_ang values must be positive.")
    ready = len(deltas) >= MIN_PLATEAU_DELTA_COUNT
    return {
        "delta_ang_values": deltas,
        "delta_count": len(deltas),
        "min_delta_count_for_plateau": MIN_PLATEAU_DELTA_COUNT,
        "delta_plateau_ready": ready,
        "delta_sweep_status": "plateau_capable" if ready else "insufficient_for_plateau",
        "delta_selection_policy": "observed_plateau_only",
    }


def select_delta_from_plateau(
    measurements: Mapping[float, float] | Iterable[tuple[float, float]],
    *,
    rel_tol: float = 0.05,
) -> dict[str, Any]:
    """Pick the operating ``delta`` from *measured* values, or refuse to pick.

    ``measurements`` maps each swept ``delta`` to one scalar summary of the
    resulting derivative (a Frobenius norm, an element, an error). The plateau
    is the longest run of consecutive amplitudes whose values agree within
    ``rel_tol`` of the run median; the selected amplitude is the *largest* one
    inside it, i.e. the best signal against the FD noise floor while truncation
    error is still flat.

    Raises when the sweep is too short or no run holds: there is no fallback to
    a preferred value, by design.
    """
    items = sorted(dict(measurements).items())
    if len(items) < MIN_PLATEAU_DELTA_COUNT:
        raise FdPerturbationError(
            f"A plateau needs at least {MIN_PLATEAU_DELTA_COUNT} deltas; got {len(items)}."
        )
    deltas = [float(delta) for delta, _ in items]
    values = [float(value) for _, value in items]
    if any(delta <= 0 for delta in deltas) or not all(np.isfinite(values)):
        raise FdPerturbationError("Plateau measurements must be positive deltas with finite values.")

    def spread(start: int, stop: int) -> float:
        window = values[start:stop]
        reference = abs(float(np.median(window)))
        if reference < MIN_DIRECTION_NORM:
            return float("inf")
        return float(max(window) - min(window)) / reference

    best: tuple[int, int] | None = None
    for start in range(len(items)):
        for stop in range(start + 2, len(items) + 1):
            if spread(start, stop) > rel_tol:
                continue
            length = stop - start
            if best is None or length > best[1] - best[0] or (
                length == best[1] - best[0] and spread(start, stop) < spread(*best)
            ):
                best = (start, stop)
    if best is None:
        raise FdPerturbationError(
            f"No plateau within rel_tol={rel_tol:g}: the derivative never stabilises over {deltas}."
        )
    start, stop = best
    return {
        "selected_delta_ang": deltas[stop - 1],
        "plateau_delta_ang": deltas[start:stop],
        "plateau_values": values[start:stop],
        "plateau_relative_spread": spread(start, stop),
        "rel_tol": float(rel_tol),
        "delta_ang_values": deltas,
        "criterion": (
            "longest run of consecutive deltas within rel_tol of the run median; "
            "largest delta of that run is selected"
        ),
    }


# --------------------------------------------------------------------------- #
# Neighbour topology
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TopologyMargin:
    """How close ``R +- delta v`` comes to changing the neighbour list."""

    preserved: bool
    candidate_pair_count: int
    neighbor_pair_count: int
    min_margin_ang: float
    min_margin_pair: tuple[int, int, tuple[int, int, int]] | None
    max_pair_distance_change_ang: float
    crossing_pairs: tuple[dict[str, Any], ...]
    cutoff_ang: float
    cutoff_mode: str
    geometries: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_preserved": self.preserved,
            "candidate_pair_count": self.candidate_pair_count,
            "neighbor_pair_count": self.neighbor_pair_count,
            "min_margin_ang": self.min_margin_ang,
            "min_margin_pair": list(self.min_margin_pair[:2]) + [list(self.min_margin_pair[2])]
            if self.min_margin_pair
            else None,
            "max_pair_distance_change_ang": self.max_pair_distance_change_ang,
            "crossing_pairs": [dict(entry) for entry in self.crossing_pairs],
            "cutoff_ang": self.cutoff_ang,
            "cutoff_mode": self.cutoff_mode,
            "geometries": list(self.geometries),
        }


def atom_cutoff_radii_ang(orbital_contract: Mapping[str, Any]) -> list[float]:
    """Per-atom PAO radius in Ang from an ``orbital_contract_v1`` payload.

    The neighbour rule of a SIESTA sparse block is ``d_ij <= rc_i + rc_j``, so
    the per-atom radius is the largest orbital cutoff of its species.
    """
    bohr_to_ang = 0.529177210903
    body = orbital_contract.get("basis") or orbital_contract
    radii_by_label: dict[str, float] = {}
    for species in body.get("species") or []:
        cutoffs = [float(orbital.get("cutoff_bohr", "nan")) for orbital in species.get("orbitals") or []]
        finite = [value for value in cutoffs if np.isfinite(value)]
        if not finite:
            raise FdPerturbationError(f"Species {species.get('label')!r} declares no finite orbital cutoff.")
        radii_by_label[str(species["label"])] = max(finite) * bohr_to_ang
    atoms = orbital_contract.get("atoms") or []
    if not atoms:
        raise FdPerturbationError("Orbital contract carries no atom list to assign cutoffs to.")
    try:
        return [radii_by_label[str(atom["species_label"])] for atom in atoms]
    except KeyError as exc:
        raise FdPerturbationError(f"No basis cutoff for species {exc!s} in the orbital contract.") from exc


def _periodic_images(
    lattice_vectors_ang: Any, search_radius_ang: float
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Lattice images that can bring a neighbour within ``search_radius_ang``.

    Returns the cell indices and their translations, in the same order. Only the
    lexicographically non-negative half is enumerated: ``d(i, j+T)`` equals
    ``d(j, i-T)``, so the other half would only duplicate pairs.
    """
    if lattice_vectors_ang is None:
        return [(0, 0, 0)], np.zeros((1, 3), dtype=np.float64)
    cell = np.asarray(lattice_vectors_ang, dtype=np.float64)
    if cell.shape != (3, 3):
        raise FdPerturbationError(f"Lattice vectors must have shape (3, 3); got {cell.shape}.")
    volume = abs(float(np.linalg.det(cell)))
    if volume < MIN_DIRECTION_NORM:
        raise FdPerturbationError("Lattice vectors are singular; cannot enumerate periodic images.")
    ranges = []
    for index in range(3):
        other = cell[[i for i in range(3) if i != index]]
        # Distance between the two planes spanned by the other two vectors.
        width = volume / float(np.linalg.norm(np.cross(other[0], other[1])))
        limit = int(np.ceil(search_radius_ang / width))
        ranges.append(range(-limit, limit + 1))
    indices = [n for n in itertools.product(*ranges) if n >= (0, 0, 0)]
    return indices, np.asarray(indices, dtype=np.float64) @ cell


def topology_margin(
    positions_ang: Sequence[Sequence[float]],
    *,
    direction: Direction,
    delta_ang: float,
    cutoff_ang: float | Sequence[float],
    lattice_vectors_ang: Any = None,
    method: str = "central",
) -> TopologyMargin:
    """Compare the neighbour list of ``R`` with those of ``R +- delta v``.

    ``cutoff_ang`` is either one pair cutoff for the whole system (a Graph2Mat
    edge cutoff) or one radius per atom, in which case the pair cutoff is
    ``r_i + r_j`` (the SIESTA PAO rule). Every pair that could possibly cross is
    enumerated: a pair further than ``cutoff + 2 delta max|v_i|`` in ``R`` cannot
    reach the cutoff under the displacement.
    """
    positions = np.asarray(positions_ang, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise FdPerturbationError(f"positions must have shape (N, 3); got {positions.shape}.")
    if positions.shape[0] != direction.atom_count:
        raise FdPerturbationError(
            f"Direction {direction.name!r} spans {direction.atom_count} atoms but the structure has {positions.shape[0]}."
        )
    delta = float(delta_ang)
    if delta <= 0:
        raise FdPerturbationError("delta_ang must be positive.")

    radii = np.asarray(cutoff_ang, dtype=np.float64).reshape(-1)
    if radii.size == 1:
        cutoff_mode = "pair_cutoff"
        pair_cutoff = lambda i, j: float(radii[0])  # noqa: E731
        max_cutoff = float(radii[0])
    elif radii.size == positions.shape[0]:
        cutoff_mode = "per_atom_radius_sum"
        pair_cutoff = lambda i, j: float(radii[i] + radii[j])  # noqa: E731
        max_cutoff = float(2.0 * radii.max())
    else:
        raise FdPerturbationError(
            f"cutoff_ang must be a scalar or one radius per atom ({positions.shape[0]}); got {radii.size} values."
        )
    if max_cutoff <= 0:
        raise FdPerturbationError("cutoff_ang must be positive.")

    signs = signs_for_method(method)
    geometries: dict[str, np.ndarray] = {"base": positions}
    for sign in signs:
        geometries["plus" if sign > 0 else "minus"] = positions + sign * delta * direction.vectors

    slack = 2.0 * delta * direction.max_atom_norm
    search_radius = max_cutoff + slack + 1e-9
    image_indices, translations = _periodic_images(lattice_vectors_ang, search_radius)

    base_tree = cKDTree(positions)
    candidates: list[tuple[int, int, int]] = []
    for image_index, translation in enumerate(translations):
        shifted = cKDTree(positions + translation)
        for i, neighbours in enumerate(base_tree.query_ball_tree(shifted, search_radius)):
            for j in neighbours:
                if image_indices[image_index] == (0, 0, 0) and j <= i:
                    continue
                candidates.append((i, j, image_index))

    min_margin = float("inf")
    min_margin_pair: tuple[int, int, tuple[int, int, int]] | None = None
    max_change = 0.0
    neighbor_pairs = 0
    crossings: list[dict[str, Any]] = []
    for i, j, image_index in candidates:
        translation = translations[image_index]
        cutoff = pair_cutoff(i, j)
        distances = {
            role: float(np.linalg.norm(geometry[j] + translation - geometry[i]))
            for role, geometry in geometries.items()
        }
        membership = {role: distance <= cutoff for role, distance in distances.items()}
        for role, distance in distances.items():
            margin = abs(distance - cutoff)
            if margin < min_margin:
                min_margin = margin
                min_margin_pair = (i, j, image_indices[image_index])
        base_distance = distances["base"]
        max_change = max(max_change, max(abs(distance - base_distance) for distance in distances.values()))
        if membership["base"]:
            neighbor_pairs += 1
        if len(set(membership.values())) > 1:
            crossings.append(
                {
                    "atom_i_zero_based": i,
                    "atom_j_zero_based": j,
                    "image": list(image_indices[image_index]),
                    "cutoff_ang": cutoff,
                    "distances_ang": distances,
                    "in_neighbor_list": membership,
                }
            )

    return TopologyMargin(
        preserved=not crossings,
        candidate_pair_count=len(candidates),
        neighbor_pair_count=neighbor_pairs,
        min_margin_ang=float(min_margin) if candidates else float("inf"),
        min_margin_pair=min_margin_pair,
        max_pair_distance_change_ang=float(max_change),
        crossing_pairs=tuple(crossings),
        cutoff_ang=max_cutoff,
        cutoff_mode=cutoff_mode,
        geometries=tuple(sorted(geometries)),
    )


def assert_topology_preserved(margin: TopologyMargin, *, context: str = "") -> TopologyMargin:
    """Refuse a perturbation that moves a pair across the neighbour cutoff."""
    if margin.preserved:
        return margin
    first = margin.crossing_pairs[0]
    where = f" ({context})" if context else ""
    raise FdPerturbationError(
        f"Perturbation changes the neighbour topology{where}: {len(margin.crossing_pairs)} pair(s) cross the cutoff, "
        f"first atoms {first['atom_i_zero_based']}-{first['atom_j_zero_based']} image {first['image']} "
        f"at cutoff {first['cutoff_ang']:.6g} Ang with distances {first['distances_ang']}. "
        "A derivative across a graph discontinuity is not a derivative."
    )


def demo() -> None:
    """Self-check: one-hot exactness, translation invariance, topology gate."""
    positions = [[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]]
    cell = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]

    coordinate = one_hot(2, 1, "y")
    assert displace(positions, coordinate, 0.1)[1] == [1.42, 0.1, 0.0]
    assert displace(positions, coordinate, 0.1)[0] == positions[0]

    translation = uniform_translation(2, "x")
    moved = displace(positions, translation, 0.1)
    assert abs((moved[1][0] - moved[0][0]) - (positions[1][0] - positions[0][0])) < 1e-12

    assert abs(np.linalg.norm(random_collective(2, seed=3).vectors) - 1.0) < 1e-12
    assert delta_sweep(0.01) == (0.005, 0.01, 0.02)
    assert select_delta_from_plateau({0.005: 1.0, 0.01: 1.01, 0.02: 1.4})["selected_delta_ang"] == 0.01
    assert not delta_sweep_status([0.01])["delta_plateau_ready"]

    # 2.0 Ang pair cutoff with atoms 1.42 Ang apart: safe at 0.1, crossed at 0.6.
    safe = topology_margin(positions, direction=coordinate, delta_ang=0.1, cutoff_ang=2.0, lattice_vectors_ang=cell)
    assert safe.preserved and safe.neighbor_pair_count == 1
    crossed = topology_margin(
        positions, direction=one_hot(2, 1, "x"), delta_ang=0.6, cutoff_ang=2.0, lattice_vectors_ang=cell
    )
    assert not crossed.preserved
    print("fd_perturbation_space: ok")


if __name__ == "__main__":
    demo()
