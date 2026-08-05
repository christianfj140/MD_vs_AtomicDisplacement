#!/usr/bin/env python3
"""Atomistic p_z tight binding for the rigid (31,30) twisted bilayer graphene cell.

Independent, cheap and *explicitly approximate* reference for the Graph2Mat bands.
It is NOT a DFT ground truth: one p_z orbital per carbon, S = I, universal published
parameters applied to the exact FDF coordinates without any recalibration.

Model: P. Moon and M. Koshino, Phys. Rev. B 87, 205404 (2013), Slater-Koster
interpolation between pi and sigma bonds with a single exponential decay length.

Stages (each writes its own stages/*.json, then `aggregate` builds the UI contract):
    --self-test   small-cell physics + numerics asserts (fast, no target cell)
    --preflight   target cell: hermiticity, sigma anchoring, K/K', cutoff convergence
    --neutrality  E_F by half filling on a Gamma-centred mesh (inertia counting)
    --bands       the same 31-point K-Gamma-M-K path used by the Graph2Mat production
    --dos         partial DOS of the low-energy window on the same mesh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

# Apply the real thread ceiling before NumPy/SciPy load their BLAS runtimes. Four is the
# conservative default for this CPU; callers may request fewer, never more than eight.
DEFAULT_SOLVER_THREADS = 4
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    try:
        _requested_threads = int(os.environ.get(_thread_variable, DEFAULT_SOLVER_THREADS))
    except ValueError:
        _requested_threads = DEFAULT_SOLVER_THREADS
    os.environ[_thread_variable] = str(
        max(1, min(DEFAULT_SOLVER_THREADS, _requested_threads))
    )

import numpy as np
import sisl
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from scipy.spatial import cKDTree

from run_deeph_sparse_spectrum import cpu_package_temperature_c, free_disk_percent

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "Comparison/results/tbg_tight_binding"
TARGET_FDF = REPO / "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf"
MONOLAYER_FDF = REPO / "materials/bilayer_graphene_AA/RUN.fdf"
MINIMUM_FREE_DISK_PERCENT = 12.0
CPU_WARNING_C = 80.0
CPU_STOP_C = 82.0
CPU_COOLDOWN_C = 75.0
SOLVER_THREADS = max(
    int(os.environ[name])
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
)

# --- Moon-Koshino parameters, frozen. Universal values of the published model; the
# --- reference distances stay at their published values (a = 2.46 Ang) and are NOT
# --- rescaled to this geometry's a = 2.48 Ang. Applying the published radial function
# --- to d_CC = 1.4318 Ang therefore gives V_pppi = -2.63 eV rather than -2.70 eV,
# --- which is a physical consequence of the dilated cell, not a free parameter.
V_PP_PI_0 = -2.7  # eV
V_PP_SIGMA_0 = 0.48  # eV
A_REFERENCE = 2.46  # Ang
A_CARBON_CARBON = A_REFERENCE / math.sqrt(3.0)  # 1.4202817 Ang
D_INTERLAYER = 3.35  # Ang
DECAY_LENGTH = 0.184 * A_REFERENCE  # 0.45264 Ang
CUTOFF = 4.0 * A_CARBON_CARBON  # 5.6811266 Ang, hard truncation as published

# Bump when a stage's numerical procedure changes in a way that invalidates cached
# results. It enters every stage contract, so stale artifacts stop being reused.
ALGORITHM_VERSION = 3

SPIN_DEGENERACY = 2
BAND_STATES = 32
MESH_STATES = 64
NEUTRALITY_MESH = 12
DOS_BROADENING_EV = 0.002
DOS_WINDOW_EV = 0.5

MODEL_PROVENANCE = {
    "model_id": "moon_koshino_2013_literal_rigid",
    "reference": "P. Moon and M. Koshino, Phys. Rev. B 87, 205404 (2013)",
    "hopping": "h(d) = V_pppi(d) (1 - (dz/d)^2) + V_ppsigma(d) (dz/d)^2",
    "radial": "V_pppi(d) = V_pppi0 exp(-(d - a_cc)/delta0); V_ppsigma(d) = V_ppsigma0 exp(-(d - d0)/delta0)",
    "sign_convention": "matrix element directly; intralayer first neighbour is negative",
    "v_pp_pi_0_eV": V_PP_PI_0,
    "v_pp_sigma_0_eV": V_PP_SIGMA_0,
    "a_reference_ang": A_REFERENCE,
    "a_carbon_carbon_ang": A_CARBON_CARBON,
    "d_interlayer_ang": D_INTERLAYER,
    "decay_length_ang": DECAY_LENGTH,
    "cutoff_ang": CUTOFF,
    "cutoff_shape": "hard",
    "onsite_eV": 0.0,
    "overlap": "identity",
    "recalibrated_to_this_geometry": False,
    "scope": "atomistic p_z reference, approximate; not a DFT ground truth",
}

RESOURCE_OBSERVATIONS = {
    "initial_cpu_temperature_c": cpu_package_temperature_c(),
    "maximum_cpu_temperature_c": cpu_package_temperature_c(),
    "minimum_free_disk_percent": None,
    "solver_threads": SOLVER_THREADS,
}


class ResourceGuardError(RuntimeError):
    """Safe, resumable stop caused by thermal or disk headroom."""


def guard_resources(context: str, *, cooldown: bool = True) -> dict:
    """Enforce the 12 % disk and 82 C preventive CPU limits between solves."""
    free = free_disk_percent(RESULTS if RESULTS.exists() else REPO)
    previous_free = RESOURCE_OBSERVATIONS["minimum_free_disk_percent"]
    RESOURCE_OBSERVATIONS["minimum_free_disk_percent"] = (
        free if previous_free is None else min(float(previous_free), free)
    )
    if free <= MINIMUM_FREE_DISK_PERCENT:
        raise ResourceGuardError(
            f"{context}: disk headroom {free:.2f}% is at/below the "
            f"{MINIMUM_FREE_DISK_PERCENT:.0f}% preventive limit"
        )

    temperature = cpu_package_temperature_c()
    if temperature is None:
        raise ResourceGuardError(
            f"{context}: CPU package temperature is unavailable; refusing an unmonitored run"
        )
    if temperature is not None:
        previous_temperature = RESOURCE_OBSERVATIONS["maximum_cpu_temperature_c"]
        RESOURCE_OBSERVATIONS["maximum_cpu_temperature_c"] = max(
            float(previous_temperature or temperature), temperature
        )
    if temperature is not None and temperature >= CPU_STOP_C:
        raise ResourceGuardError(
            f"{context}: CPU temperature {temperature:.1f} C reached the "
            f"{CPU_STOP_C:.0f} C preventive stop"
        )
    if cooldown and temperature is not None and temperature >= CPU_WARNING_C:
        while temperature is not None and temperature > CPU_COOLDOWN_C:
            print(
                f"[{context}] CPU {temperature:.1f} C: solver already limited to "
                f"{SOLVER_THREADS} threads; cooling to {CPU_COOLDOWN_C:.0f} C",
                flush=True,
            )
            time.sleep(15)
            temperature = cpu_package_temperature_c()
            if temperature is not None:
                RESOURCE_OBSERVATIONS["maximum_cpu_temperature_c"] = max(
                    float(RESOURCE_OBSERVATIONS["maximum_cpu_temperature_c"] or temperature),
                    temperature,
                )
            if temperature is not None and temperature >= CPU_STOP_C:
                raise ResourceGuardError(
                    f"{context}: CPU temperature {temperature:.1f} C reached the "
                    f"{CPU_STOP_C:.0f} C preventive stop while cooling"
                )
    return {
        "cpu_temperature_c": temperature,
        "free_disk_percent": free,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(temporary: Path, path: Path) -> None:
    os.replace(temporary, path)
    # Without syncing the directory the rename itself can be lost on power failure,
    # leaving the file absent even though its contents were flushed.
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            # allow_nan=False: NaN/Infinity are not valid JSON. Emitting them turns a
            # numerical failure upstream into a file that strict parsers reject, which is
            # far better than silently publishing a NaN observable.
            json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def hopping(
    distance: np.ndarray, dz: np.ndarray, *, a_reference: float = A_REFERENCE
) -> np.ndarray:
    """Slater-Koster p_z-p_z matrix element in eV."""
    a_carbon_carbon = a_reference / math.sqrt(3.0)
    decay_length = 0.184 * a_reference
    projection = (dz / distance) ** 2
    pi_term = V_PP_PI_0 * np.exp(-(distance - a_carbon_carbon) / decay_length)
    sigma_term = V_PP_SIGMA_0 * np.exp(-(distance - D_INTERLAYER) / decay_length)
    return pi_term * (1.0 - projection) + sigma_term * projection


class PzTightBinding:
    """H(k) for one p_z orbital per atom, orthogonal basis, 2D Bloch phases."""

    def __init__(
        self,
        geometry: sisl.Geometry,
        cutoff: float | None = None,
        *,
        a_reference: float = A_REFERENCE,
    ):
        self.a_reference = float(a_reference)
        self.cutoff = float(cutoff if cutoff is not None else 4 * a_reference / math.sqrt(3))
        self.cell = np.asarray(geometry.cell, dtype=float)
        self.xyz, self.wrap_shift_max_ang = self._wrap_in_plane(
            np.asarray(geometry.xyz, dtype=float)
        )
        self.count = len(self.xyz)
        self._build_bonds()

    def _wrap_in_plane(self, xyz: np.ndarray) -> tuple[np.ndarray, float]:
        """Fold the in-plane coordinates into the cell.

        The periodic reach in _build_bonds assumes every atom lies inside the cell. An
        atom translated out by a lattice vector describes the same crystal, but with the
        moire reach of [1, 1] it silently loses neighbours: displacing one atom by -a2 in
        a reach-1 cell drops 56 bonds and moves the spectrum by 1.4 eV. Folding first
        makes the model exactly invariant under per-atom lattice translations.
        """
        plane = self.cell[:2, :2]
        fractional = np.linalg.solve(plane.T, xyz[:, :2].T).T
        # Subtract only the integer part, so an atom already inside the cell keeps its
        # coordinates bit-for-bit. Rebuilding every position from fractionals instead
        # perturbs them by ~1e-14 Ang, which is enough to flip a marginal SuperLU pivot
        # and lose the inertia certification.
        translation = np.floor(fractional) @ plane
        wrapped = np.column_stack([xyz[:, :2] - translation, xyz[:, 2]])
        return wrapped, float(np.max(np.linalg.norm(wrapped - xyz, axis=1)))

    def _build_bonds(self) -> None:
        area = abs(float(np.cross(self.cell[0], self.cell[1])[2]))
        # Distance between the lattice lines n_i = const is area / |a_(1-i)|.
        reach = [
            int(math.ceil(self.cutoff * float(np.linalg.norm(self.cell[1 - i])) / area))
            for i in (0, 1)
        ]
        shifts = [
            (n1, n2)
            for n1 in range(-reach[0], reach[0] + 1)
            for n2 in range(-reach[1], reach[1] + 1)
        ]
        replicas = np.concatenate(
            [self.xyz + n1 * self.cell[0] + n2 * self.cell[1] for n1, n2 in shifts]
        )
        replica_atom = np.tile(np.arange(self.count), len(shifts))
        replica_image = np.repeat(np.asarray(shifts, dtype=np.int64), self.count, axis=0)

        neighbours = cKDTree(self.xyz).query_ball_tree(cKDTree(replicas), self.cutoff)
        rows = np.repeat(np.arange(self.count), [len(item) for item in neighbours])
        flat = np.concatenate(
            [np.asarray(item, dtype=np.int64) for item in neighbours]
            or [np.zeros(0, dtype=np.int64)]
        )
        delta = replicas[flat] - self.xyz[rows]
        distance = np.linalg.norm(delta, axis=1)
        # Drop only the on-site term itself; i == j with a non-zero translation is a
        # legitimate hopping in small cells and must survive.
        keep = distance > 1e-8
        rows, flat, delta, distance = rows[keep], flat[keep], delta[keep], distance[keep]

        self.minimum_distance = float(distance.min()) if distance.size else float("nan")
        self.bond_count = int(distance.size)
        self.periodic_reach = reach
        values = hopping(distance, delta[:, 2], a_reference=self.a_reference)

        # The on-site entries are stored explicitly (value 0) so that the shifted matrix
        # always has a structural diagonal: SuperLU needs it to pivot on the diagonal,
        # which is what makes the inertia count below valid.
        diagonal = np.arange(self.count)
        self.rows = np.concatenate([rows, diagonal])
        self.cols = np.concatenate([replica_atom[flat], diagonal])
        self.image = np.concatenate(
            [replica_image[flat], np.zeros((self.count, 2), dtype=np.int64)]
        )
        self.hop = np.concatenate([values, np.zeros(self.count)])

    def hk(self, k, shift: float = 0.0) -> csr_matrix:
        phase = np.exp(2j * np.pi * (self.image @ np.asarray(k, dtype=float)[:2]))
        matrix = csr_matrix(
            (self.hop * phase, (self.rows, self.cols)),
            shape=(self.count, self.count),
            dtype=complex,
        )
        if shift:
            matrix.setdiag(matrix.diagonal() - shift)
        return matrix

    def hermiticity_error(self, k) -> float:
        matrix = self.hk(k)
        return float(abs(matrix - matrix.getH()).max())

    def solve(self, k, sigma: float, count: int, vectors: bool = False) -> dict:
        """Shift-invert window around sigma plus the exact number of states below it.

        One LU factorisation serves both: the inertia (Sylvester, valid because
        SuperLU is forced to pivot on the diagonal) and the ARPACK OPinv operator.
        """
        matrix = self.hk(k)
        factorisation = splu(
            self.hk(k, shift=sigma).tocsc(),
            diag_pivot_thresh=0.0,
            permc_spec="MMD_AT_PLUS_A",
            options=dict(SymmetricMode=True, Equil=False),
        )
        # SuperLU is a general LU solver. Its FAQ only supports reading inertia from
        # diag(U) in symmetric mode when no off-diagonal row pivoting occurred. Reject
        # the count otherwise instead of silently inventing absolute band indices.
        symmetric_permutation = bool(
            np.array_equal(factorisation.perm_r, factorisation.perm_c)
        )
        diagonal_u = factorisation.U.diagonal()
        maximum_pivot_imaginary = float(np.max(np.abs(diagonal_u.imag)))
        relative_pivot_imaginary = maximum_pivot_imaginary / max(
            float(np.max(np.abs(diagonal_u))), np.finfo(float).eps
        )
        # What the inertia actually reads is sign(Re d_i), one pivot at a time. Comparing
        # the imaginary part against the LARGEST pivot says nothing about whether a given
        # sign is determinate: in this system max|d| = 4.5e4 while the smallest |Re d| is
        # 2.0e-4, so a global ratio mixes scales that differ by nine orders of magnitude.
        # It rejected 2 of 144 production k points whose counts were then confirmed
        # correct against dense diagonalisation (both gave 5580). The certified quantity
        # is therefore the per-pivot margin |Im d_i| / |Re d_i|: in exact arithmetic D is
        # real, so |Im d_i| measures the local roundoff on the very number whose sign is
        # being read. Observed worst case across the mesh is 1.5e-6, six orders below
        # ambiguity; genuine LDL^H breakdown drives this ratio towards 1.
        pivot_sign_margin = float(
            np.max(np.abs(diagonal_u.imag) / np.maximum(np.abs(diagonal_u.real), np.finfo(float).tiny))
        )
        if not symmetric_permutation or pivot_sign_margin > 1e-3:
            raise RuntimeError(
                "SuperLU inertia is not certified: off-diagonal pivoting or complex "
                f"D detected (symmetric_permutation={symmetric_permutation}, "
                f"pivot_sign_margin={pivot_sign_margin:.3e}, "
                f"relative_imag={relative_pivot_imaginary:.3e})"
            )
        below = int((diagonal_u.real < 0).sum())
        operator = LinearOperator(
            matrix.shape, matvec=factorisation.solve, dtype=complex
        )
        count = min(count, self.count - 2)
        result = eigsh(
            matrix,
            k=count,
            sigma=sigma,
            OPinv=operator,
            ncv=min(self.count, max(4 * count + 20, 80)),
            return_eigenvectors=vectors,
        )
        if vectors:
            energies, states = result
        else:
            energies, states = result, None
        order = np.argsort(energies)
        energies = np.asarray(energies)[order].real
        if states is not None:
            states = np.asarray(states)[:, order]
            residual = float(
                np.max(
                    np.linalg.norm(matrix @ states - states * energies, axis=0)
                    / np.linalg.norm(states, axis=0)
                )
            )
        else:
            residual = float("nan")
        # Absolute band index of energies[j] is `first_index + j`.
        first_index = below - int((energies < sigma).sum())
        return {
            "energies": energies,
            "states": states,
            "states_below_sigma": below,
            "first_index": first_index,
            "inertia_diagnostic": {
                "method": "superlu_symmetric_mode_diag_u_no_offdiagonal_pivot",
                "symmetric_row_column_permutation": symmetric_permutation,
                "maximum_pivot_imaginary": maximum_pivot_imaginary,
                "relative_pivot_imaginary": relative_pivot_imaginary,
                "pivot_sign_margin": pivot_sign_margin,
                "pivot_sign_margin_threshold": 1e-3,
                "smallest_absolute_real_pivot": float(np.min(np.abs(diagonal_u.real))),
                "certification": (
                    "per-pivot |Im d|/|Re d|; cross-checked against dense diagonalisation "
                    "at the two mesh points with the largest imaginary contamination"
                ),
                "status": "empirically_validated_not_sparse_ldlh_proof",
            },
            "residual_eV": residual,
        }


def read_geometry(path: Path) -> sisl.Geometry:
    return sisl.get_sile(str(path)).read_geometry()


def layer_labels(xyz: np.ndarray) -> np.ndarray:
    """0 = lower layer, 1 = upper layer, split at the mid-point of the two z clusters."""
    z = xyz[:, 2]
    boundary = 0.5 * (z.min() + z.max())
    return (z > boundary).astype(int)


def reciprocal(cell: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(cell).T


def corner_k(cell: np.ndarray) -> tuple[float, float]:
    """Fractional BZ corner K. Depends on the sign convention of the cell vectors:
    a 120 deg direct cell (the primitive graphene cells here) puts K at (1/3, 1/3),
    a 60 deg direct cell (the moire supercell) puts it at (1/3, 2/3)."""
    a1, a2 = cell[0, :2], cell[1, :2]
    cosine = float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))
    return (1 / 3, 1 / 3) if cosine < 0 else (1 / 3, 2 / 3)


def band_path(cell: np.ndarray, per_segment: int = 11) -> dict:
    """K -> Gamma -> M -> K, identical to the persisted Graph2Mat production path."""
    k_corner = corner_k(cell)
    nodes = [
        (np.array([k_corner[0], k_corner[1], 0.0]), "K"),
        (np.array([0.0, 0.0, 0.0]), "Γ"),
        (np.array([0.5, 0.5, 0.0]), "M"),
        (np.array([k_corner[0], k_corner[1], 0.0]), "K"),
    ]
    basis = reciprocal(cell)
    points: list[np.ndarray] = []
    labels: list[str] = []
    for index in range(len(nodes) - 1):
        start, start_label = nodes[index]
        end, end_label = nodes[index + 1]
        for step in range(per_segment):
            if index and step == 0:
                continue  # drop the duplicated segment endpoint
            fraction = step / (per_segment - 1)
            points.append(start + fraction * (end - start))
            labels.append(
                start_label if step == 0 else (end_label if step == per_segment - 1 else "")
            )
    frac = np.asarray(points)
    cartesian = frac @ basis
    distance = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
    )
    return {
        "fractional": frac,
        "labels": labels,
        "distance": distance,
        "contract": {
            "coordinate_system": "reciprocal_lattice_vectors",
            "coordinates": [
                {"k": list(map(float, node)), "label": label} for node, label in nodes
            ],
            "direct_lattice_vectors_ang": cell.tolist(),
            "name": "moire_k-gamma-m-k",
            "points_per_segment": per_segment,
            "reciprocal_lattice_vectors_inv_ang": basis.tolist(),
            "sample_count": len(frac),
        },
    }


def gamma_mesh(size: int) -> np.ndarray:
    grid = np.arange(size) / size
    return np.array([[a, b, 0.0] for a in grid for b in grid])


def monolayer_geometry() -> sisl.Geometry:
    """Lower layer of the AA bilayer cell: same lattice constant as the target."""
    geometry = read_geometry(MONOLAYER_FDF)
    xyz = np.asarray(geometry.xyz)
    keep = np.where(xyz[:, 2] < 0.5 * (xyz[:, 2].min() + xyz[:, 2].max()))[0]
    return geometry.sub(keep)


def dirac_level(
    geometry: sisl.Geometry | None = None,
    cutoff: float | None = None,
    *,
    a_reference: float = A_REFERENCE,
) -> dict:
    """Monolayer Dirac energy and Fermi velocity with this exact parametrisation."""
    geometry = geometry if geometry is not None else monolayer_geometry()
    model = PzTightBinding(geometry, cutoff=cutoff, a_reference=a_reference)
    cell = np.asarray(geometry.cell)
    k_corner = np.array([*corner_k(cell), 0.0])
    at_k = np.linalg.eigvalsh(model.hk(k_corner).toarray())
    energy = float(0.5 * (at_k[0] + at_k[1]))
    basis = reciprocal(cell)
    delta = 1e-4
    direction = np.array([1.0, 0.0, 0.0])
    step = delta * np.linalg.solve(basis.T, direction)
    nearby = np.linalg.eigvalsh(model.hk(k_corner + step).toarray())
    # hbar * v_F in eV*Ang; divide by hbar to get m/s.
    slope = float((nearby[1] - nearby[0]) / (2.0 * delta))
    return {
        "dirac_energy_eV": energy,
        "degeneracy_splitting_eV": float(abs(at_k[1] - at_k[0])),
        "hbar_v_fermi_eV_ang": slope,
        "v_fermi_m_per_s": slope * 1.602176634e-19 / 1.054571817e-34 * 1e-10,
        "lattice_constant_ang": float(np.linalg.norm(cell[0])),
        "k_corner_fractional": [float(value) for value in k_corner],
    }


# --------------------------------------------------------------------------------------
# stages


def moire_fermi_velocity(
    model: "PzTightBinding", sigma: float, occupied: int, states: int = BAND_STATES
) -> dict:
    """Measured renormalisation of the Dirac velocity at the moire K point.

    This is the magic-angle diagnostic, and it settles by measurement a question that is
    otherwise argued from citations: whether THIS geometry sits at the magic angle of THIS
    parametrisation. At a magic angle the moire Dirac velocity collapses, v*/v_F -> 0.
    Being off the magic angle by ~10% in twist leaves v*/v_F at the tens-of-percent level,
    so the ratio discriminates sharply without appealing to any published angle.
    """
    basis = reciprocal(model.cell)
    corner = np.array([*corner_k(model.cell), 0.0])
    gamma_k = float(np.linalg.norm(corner @ basis))
    samples = []
    for fraction in (0.01, 0.02):
        k = corner + fraction * (np.zeros(3) - corner)  # towards Gamma
        energies = indexed_energies(
            model.solve(k, sigma, states), np.arange(occupied - 2, occupied + 2)
        )
        splitting = float(np.mean(energies[2:]) - np.mean(energies[:2]))
        delta_k = fraction * gamma_k
        samples.append(
            {
                "fraction_of_gamma_k": fraction,
                "delta_k_inv_ang": delta_k,
                "splitting_eV": splitting,
                "hbar_v_star_eV_ang": splitting / (2.0 * delta_k),
            }
        )
    # Linear extrapolation of the two smallest samples to delta_k -> 0.
    first, second = samples
    slope = (first["hbar_v_star_eV_ang"] - second["hbar_v_star_eV_ang"]) / (
        first["delta_k_inv_ang"] - second["delta_k_inv_ang"]
    )
    extrapolated = first["hbar_v_star_eV_ang"] - slope * first["delta_k_inv_ang"]
    monolayer = dirac_level()["hbar_v_fermi_eV_ang"]
    ratio = extrapolated / monolayer
    return {
        "samples": samples,
        "hbar_v_star_eV_ang": float(extrapolated),
        "monolayer_hbar_v_fermi_eV_ang": float(monolayer),
        "velocity_ratio_v_star_over_v_fermi": float(ratio),
        "strongly_renormalised": bool(abs(ratio) < 0.05),
        "interpretation": (
            "v*/v_F << 1 means this rigid geometry sits at (or extremely close to) the "
            "magic angle OF THIS PARAMETRISATION; it is a measurement, not a citation"
        ),
        "method": "linear fit of the central-manifold splitting along K->Gamma, extrapolated to dk=0",
    }


def guard_disk() -> dict:
    snapshot = guard_resources("write", cooldown=False)
    return {
        "free_percent": snapshot["free_disk_percent"],
        "minimum_free_percent": MINIMUM_FREE_DISK_PERCENT,
    }


def write_stage(name: str, payload: dict) -> Path:
    guard_disk()
    path = RESULTS / "stages" / f"{name}.json"
    atomic_write_json(path, payload)
    return path


def read_stage(name: str) -> dict:
    path = RESULTS / "stages" / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def geometry_fingerprint(geometry: sisl.Geometry) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(geometry.cell, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(geometry.xyz, dtype=np.float64).tobytes())
    return digest.hexdigest()


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


MESH_CACHE = "mesh_eigenvalues.npz"


def mesh_cache_signature(
    geometry: sisl.Geometry, mesh_size: int, states: int, sigma: float, occupied: int
) -> str:
    """Identity of the spectral cache: everything that determines its contents."""
    return payload_sha256(
        {
            "geometry_sha256": geometry_fingerprint(geometry),
            "model_sha256": payload_sha256(MODEL_PROVENANCE),
            "mesh_size": int(mesh_size),
            "states": int(states),
            "sigma_eV": float(sigma),
            "occupied": int(occupied),
            "algorithm_version": ALGORITHM_VERSION,
        }
    )


def load_mesh_cache(expected_signature: str | None = None) -> dict:
    """Load the spectral cache, refusing anything that is not self-identified.

    The cache is consumed by the DOS and by the aggregate mesh observables. Without a
    signature stored inside it, any NPZ with matching array shapes would be accepted.
    """
    path = RESULTS / "stages" / MESH_CACHE
    if not path.exists():
        raise SystemExit("Run --neutrality first: the mesh eigenvalue cache is missing.")
    payload = np.load(path)
    if "signature" not in payload.files:
        raise SystemExit(
            f"{MESH_CACHE} carries no signature; it predates the provenance contract. "
            "Re-run --neutrality --force to regenerate it."
        )
    signature = str(payload["signature"].item())
    if expected_signature is not None and signature != expected_signature:
        raise SystemExit(
            f"{MESH_CACHE} signature {signature[:16]} does not match the expected "
            f"{expected_signature[:16]}; re-run --neutrality --force."
        )
    eigenvalues = np.asarray(payload["eigenvalues"])
    first_indices = np.asarray(payload["first_indices"])
    if not np.all(np.isfinite(eigenvalues)) or np.any(first_indices < 0):
        raise SystemExit(f"{MESH_CACHE} contains incomplete entries; re-run --neutrality.")
    return {
        "eigenvalues": eigenvalues,
        "first_indices": first_indices,
        "signature": signature,
        "sha256": file_sha256(path),
    }


def implementation_identity() -> dict:
    identity = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    head = REPO / ".git/HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            value = (REPO / ".git" / value[5:]).read_text(encoding="utf-8").strip()
        identity["repository_commit"] = value
    except OSError:
        identity["repository_commit"] = None
    return identity


def validate_target_geometry(geometry: sisl.Geometry) -> dict:
    xyz = np.asarray(geometry.xyz, dtype=float)
    if len(xyz) != 11164:
        raise RuntimeError(f"Expected 11164 atoms in target FDF, found {len(xyz)}")
    atomic_numbers = np.asarray(geometry.atoms.Z, dtype=int)
    if not np.all(atomic_numbers == 6):
        raise RuntimeError("TBG tight binding target must contain carbon only")
    labels = layer_labels(xyz)
    populations = [int((labels == layer).sum()) for layer in (0, 1)]
    if populations != [5582, 5582]:
        raise RuntimeError(f"Expected 5582 atoms per layer, found {populations}")
    layer_z = [float(np.mean(xyz[labels == layer, 2])) for layer in (0, 1)]
    separation = layer_z[1] - layer_z[0]
    if not np.isclose(separation, D_INTERLAYER, atol=1e-8):
        raise RuntimeError(
            f"Expected {D_INTERLAYER:.8f} Ang layer separation, found {separation:.8f}"
        )
    return {
        "atoms": len(xyz),
        "atoms_per_layer": populations,
        "carbon_only": True,
        "layer_z_ang": layer_z,
        "interlayer_separation_ang": separation,
    }


def stage_contract(stage: str, **settings) -> dict:
    geometry = read_geometry(TARGET_FDF)
    contract = {
        "stage": stage,
        "algorithm_version": ALGORITHM_VERSION,
        "geometry_sha256": geometry_fingerprint(geometry),
        "model_sha256": payload_sha256(MODEL_PROVENANCE),
        "settings": settings,
    }
    contract["contract_sha256"] = payload_sha256(contract)
    return contract


def stage_reusable(name: str, contract: dict) -> bool:
    payload = read_stage(name)
    return bool(
        payload.get("status") in {"completed", "computed", "passed"}
        and payload.get("input_contract", {}).get("contract_sha256")
        == contract.get("contract_sha256")
    )


def with_runtime_metadata(payload: dict, contract: dict) -> dict:
    return {
        **payload,
        "input_contract": contract,
        "implementation": implementation_identity(),
        "resource_observations": dict(RESOURCE_OBSERVATIONS),
        "solver_threads": SOLVER_THREADS,
    }


def indexed_energies(result: dict, indices: np.ndarray) -> np.ndarray:
    local = np.asarray(indices, dtype=int) - int(result["first_index"])
    if np.any(local < 0) or np.any(local >= len(result["energies"])):
        raise RuntimeError("Shift-invert window does not contain the requested band indices")
    return np.asarray(result["energies"])[local]


def probe_observables(spectra: dict[str, dict], occupied: int) -> dict:
    central_indices = np.arange(occupied - 2, occupied + 2)
    central = np.vstack(
        [indexed_energies(spectra[label], central_indices) for label in ("Γ", "K", "M")]
    )
    below = np.array(
        [indexed_energies(spectra[label], np.array([occupied - 3]))[0] for label in ("Γ", "K", "M")]
    )
    above = np.array(
        [indexed_energies(spectra[label], np.array([occupied + 2]))[0] for label in ("Γ", "K", "M")]
    )
    valence, conduction = central[:, :2], central[:, 2:]
    return {
        "sampled_manifold_width_eV": float(central.max() - central.min()),
        "sampled_indirect_neutrality_gap_eV": float(conduction.min() - valence.max()),
        "sampled_remote_valence_gap_eV": float(valence.min() - below.max()),
        "sampled_remote_conduction_gap_eV": float(above.min() - conduction.max()),
    }


def compare_probe_spectra(
    reference: dict, candidate: dict, occupied: int, *, sign_tolerance_eV: float = 0.0
) -> dict:
    indices = np.arange(occupied - 3, occupied + 3)
    left = np.concatenate(
        [indexed_energies(reference[label], indices) for label in ("Γ", "K", "M")]
    )
    right = np.concatenate(
        [indexed_energies(candidate[label], indices) for label in ("Γ", "K", "M")]
    )
    rigid_shift = float(np.mean(right - left))
    aligned = right - rigid_shift - left
    left_observables = probe_observables(reference, occupied)
    right_observables = probe_observables(candidate, occupied)
    gap_keys = [key for key in left_observables if "gap" in key]
    return {
        "single_global_rigid_shift_eV": rigid_shift,
        "aligned_rms_eV": float(np.sqrt(np.mean(aligned**2))),
        "aligned_max_abs_eV": float(np.max(np.abs(aligned))),
        "manifold_width_change_eV": float(
            right_observables["sampled_manifold_width_eV"]
            - left_observables["sampled_manifold_width_eV"]
        ),
        # A gap smaller than the solver residual has no determinate sign, so comparing
        # np.sign on it manufactures a pass/fail out of numerical noise (the sampled
        # neutrality gap is ~5e-12 eV against a ~1e-9 eV residual). Such gaps are
        # classified as zero and reported separately instead of silently voting.
        "gap_signs_preserved": all(
            np.sign(left_observables[key]) == np.sign(right_observables[key])
            for key in gap_keys
            if max(abs(left_observables[key]), abs(right_observables[key]))
            > sign_tolerance_eV
        ),
        "gap_sign_tolerance_eV": sign_tolerance_eV,
        "gaps_below_sign_tolerance": [
            key
            for key in gap_keys
            if max(abs(left_observables[key]), abs(right_observables[key]))
            <= sign_tolerance_eV
        ],
        "reference_observables": left_observables,
        "candidate_observables": right_observables,
    }


def stage_preflight() -> dict:
    started = time.time()
    guard_resources("preflight:start")
    geometry = read_geometry(TARGET_FDF)
    geometry_contract = validate_target_geometry(geometry)
    model = PzTightBinding(geometry)
    monolayer = dirac_level()
    sigma = monolayer["dirac_energy_eV"]
    path = band_path(model.cell)
    k_corner = list(path["contract"]["coordinates"][0]["k"])
    k_prime = [k_corner[1], k_corner[0], 0.0]
    probes = {
        "Γ": [0.0, 0.0, 0.0],
        "K": k_corner,
        "M": [0.5, 0.5, 0.0],
        "K′": k_prime,
    }

    def guarded_solve(
        instance: PzTightBinding,
        label: str,
        k,
        target_sigma: float,
        *,
        vectors: bool = False,
    ) -> dict:
        guard_resources(f"preflight:{label}:before")
        result = instance.solve(k, target_sigma, BAND_STATES, vectors=vectors)
        guard_resources(f"preflight:{label}:after")
        return result

    spectra = {
        label: guarded_solve(model, label, k, sigma, vectors=True)
        for label, k in probes.items()
    }
    k_vs_kprime = float(
        np.max(np.abs(spectra["K"]["energies"] - spectra["K′"]["energies"]))
    )
    reversed_k = guarded_solve(
        model, "-K", [-value for value in k_corner], sigma
    )
    time_reversal = float(
        np.max(np.abs(spectra["K"]["energies"] - reversed_k["energies"]))
    )
    occupied = model.count // 2
    comparison_indices = np.arange(occupied - 3, occupied + 3)
    # Any observable smaller than this is numerical noise, not physics. Ten times the
    # worst eigenpair residual across the probes is a conservative floor.
    sign_tolerance = 10.0 * max(float(spectra[label]["residual_eV"]) for label in spectra)

    sigma_stability = {}
    for delta in (-0.05, 0.05):
        shifted = {
            label: guarded_solve(model, f"{label}:sigma{delta:+.2f}", k, sigma + delta)
            for label, k in probes.items()
        }
        differences = np.concatenate(
            [
                indexed_energies(shifted[label], comparison_indices)
                - indexed_energies(spectra[label], comparison_indices)
                for label in probes
            ]
        )
        sigma_stability[f"{delta:+.2f}_eV"] = {
            "max_abs_eV": float(np.max(np.abs(differences))),
            "rms_eV": float(np.sqrt(np.mean(differences**2))),
        }

    wide_cutoff = 5.0 * A_CARBON_CARBON
    wide = PzTightBinding(geometry, cutoff=wide_cutoff)
    wide_sigma = dirac_level(cutoff=wide_cutoff)["dirac_energy_eV"]
    wide_spectra = {
        label: guarded_solve(wide, f"{label}:5a0", probes[label], wide_sigma)
        for label in ("Γ", "K", "M")
    }
    cutoff_convergence = compare_probe_spectra(
        spectra, wide_spectra, occupied, sign_tolerance_eV=sign_tolerance
    )

    scaled_reference = 2.48
    scaled = PzTightBinding(geometry, a_reference=scaled_reference)
    scaled_sigma = dirac_level(a_reference=scaled_reference)["dirac_energy_eV"]
    scaled_spectra = {
        label: guarded_solve(scaled, f"{label}:scaled", probes[label], scaled_sigma)
        for label in ("Γ", "K", "M")
    }
    geometry_scaled = compare_probe_spectra(
        spectra, scaled_spectra, occupied, sign_tolerance_eV=sign_tolerance
    )
    geometry_scaled.update(
        {
            "model_id": "moon_koshino_geometry_scaled_diagnostic",
            "a_reference_ang": scaled_reference,
            "selection_role": "diagnostic_only_never_selected_by_graph2mat_agreement",
        }
    )

    sigma_maximum = max(row["max_abs_eV"] for row in sigma_stability.values())
    cutoff_passed = bool(
        cutoff_convergence["aligned_rms_eV"] < 0.002
        and abs(cutoff_convergence["manifold_width_change_eV"]) < 0.005
        and cutoff_convergence["gap_signs_preserved"]
    )
    payload = {
        "stage": "preflight",
        **geometry_contract,
        "bond_terms": int(model.bond_count),
        "cutoff_convergence_4a0_vs_5a0": cutoff_convergence,
        "cutoff_convergence_passed": cutoff_passed,
        "geometry_scaled_sensitivity": geometry_scaled,
        "geometry_sha256": geometry_fingerprint(geometry),
        "hermiticity_max_abs_error_eV": max(
            model.hermiticity_error(k) for k in probes.values()
        ),
        "inertia_diagnostic": {
            label: spectra[label]["inertia_diagnostic"] for label in spectra
        },
        "k_vs_kprime_max_abs_eV": k_vs_kprime,
        "max_residual_eV": max(float(spectra[label]["residual_eV"]) for label in spectra),
        "minimum_interatomic_distance_ang": model.minimum_distance,
        "model": MODEL_PROVENANCE,
        "moire_fermi_velocity": moire_fermi_velocity(model, sigma, occupied),
        "monolayer": monolayer,
        "in_plane_wrap_shift_max_ang": model.wrap_shift_max_ang,
        "occupied_bands_per_k": int(occupied),
        "periodic_reach": model.periodic_reach,
        "sigma_eV": sigma,
        "sigma_stability": sigma_stability,
        "states_below_sigma": {
            label: int(spectra[label]["states_below_sigma"]) for label in spectra
        },
        "time_reversal_max_abs_eV": time_reversal,
        "wall_seconds": time.time() - started,
        "window_first_index": {
            label: int(spectra[label]["first_index"]) for label in spectra
        },
    }
    payload["sigma_brackets_manifold"] = all(
        spectra[label]["first_index"] <= occupied - 3
        and spectra[label]["first_index"] + len(spectra[label]["energies"]) > occupied + 2
        for label in spectra
    )
    payload["status"] = "passed" if (
        payload["hermiticity_max_abs_error_eV"] < 1e-12
        and payload["k_vs_kprime_max_abs_eV"] < 1e-6
        and payload["time_reversal_max_abs_eV"] < 1e-8
        and payload["max_residual_eV"] < 1e-7
        and payload["sigma_brackets_manifold"]
        and sigma_maximum < 1e-7
        and cutoff_passed
    ) else "review"
    return payload


def smeared_chemical_potential(
    eigenvalues: np.ndarray,
    first_indices: np.ndarray,
    occupied: int,
    temperature_eV: float,
) -> float:
    target = float(occupied * len(eigenvalues))
    low = float(np.min(eigenvalues) - 20 * temperature_eV)
    high = float(np.max(eigenvalues) + 20 * temperature_eV)
    for _ in range(100):
        middle = 0.5 * (low + high)
        argument = np.clip((eigenvalues - middle) / temperature_eV, -700, 700)
        count = float(np.sum(first_indices)) + float(np.sum(1.0 / (1.0 + np.exp(argument))))
        if count < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def mesh_observables_from_cache(
    eigenvalues: np.ndarray, first_indices: np.ndarray, occupied: int, mesh: list[int]
) -> dict:
    required = np.arange(occupied - 3, occupied + 3)
    selected = np.empty((len(eigenvalues), len(required)), dtype=float)
    for index, (window, first) in enumerate(zip(eigenvalues, first_indices)):
        local = required - int(first)
        if np.any(local < 0) or np.any(local >= len(window)):
            return {
                "status": "unavailable",
                "reason": "cached shift-invert window does not bracket remote bands",
                "mesh": mesh,
            }
        selected[index] = window[local]
    remote_below = selected[:, 0]
    central = selected[:, 1:5]
    remote_above = selected[:, 5]
    valence, conduction = central[:, :2], central[:, 2:]
    direct_gap = np.min(conduction, axis=1) - np.max(valence, axis=1)
    return {
        "status": "computed_from_cached_eigenvalues",
        "mesh": mesh,
        "selection": "absolute_band_indices_nocc_minus_2_through_nocc_plus_1",
        "global_manifold_width_eV": float(central.max() - central.min()),
        "indirect_neutrality_gap_eV": float(conduction.min() - valence.max()),
        "minimum_direct_neutrality_gap_eV": float(direct_gap.min()),
        "remote_valence_gap_eV": float(valence.min() - remote_below.max()),
        "remote_conduction_gap_eV": float(remote_above.min() - conduction.max()),
        "convergence_status": (
            f"single_{mesh[0]}x{mesh[1]}_mesh_not_cross_mesh_converged"
        ),
    }


def stage_neutrality(mesh_size: int = NEUTRALITY_MESH, states: int = MESH_STATES) -> dict:
    started = time.time()
    guard_resources("neutrality:start")
    geometry = read_geometry(TARGET_FDF)
    validate_target_geometry(geometry)
    model = PzTightBinding(geometry)
    sigma = read_stage("preflight").get("sigma_eV") or dirac_level()["dirac_energy_eV"]
    occupied = model.count // 2
    mesh = gamma_mesh(mesh_size)
    progress_signature = payload_sha256(
        {
            "geometry_sha256": geometry_fingerprint(geometry),
            "model_sha256": payload_sha256(MODEL_PROVENANCE),
            "mesh_size": mesh_size,
            "states": states,
            "sigma_eV": float(sigma),
        }
    )

    progress_path = RESULTS / "stages" / "neutrality_progress.npz"
    if progress_path.exists():
        progress = np.load(progress_path)
        reusable_progress = (
            {"eigenvalues", "first_indices", "completed", "signature"}.issubset(progress.files)
            and tuple(progress["eigenvalues"].shape) == (len(mesh), states)
            and tuple(progress["first_indices"].shape) == (len(mesh),)
            and tuple(progress["completed"].shape) == (len(mesh),)
            and str(progress["signature"].item()) == progress_signature
        )
        if reusable_progress:
            # A matching signature only proves the file belongs to this run. Every entry
            # marked complete must also be usable: finite energies and a real window.
            done = np.asarray(progress["completed"], dtype=bool)
            rows = np.asarray(progress["eigenvalues"])[done]
            starts = np.asarray(progress["first_indices"])[done]
            reusable_progress = bool(
                np.all(np.isfinite(rows))
                and np.all(starts >= 0)
                and np.all(starts <= occupied - 1)
                and np.all(starts + states > occupied)
            )
            if not reusable_progress:
                print(
                    "[neutrality] discarding progress cache: completed entries are "
                    "incomplete or out of range",
                    flush=True,
                )
    else:
        reusable_progress = False
    if reusable_progress:
        eigenvalues_array = np.asarray(progress["eigenvalues"]).copy()
        first_indices_array = np.asarray(progress["first_indices"]).copy()
        completed = np.asarray(progress["completed"], dtype=bool).copy()
    else:
        eigenvalues_array = np.full((len(mesh), states), np.nan)
        first_indices_array = np.full(len(mesh), -1, dtype=int)
        completed = np.zeros(len(mesh), dtype=bool)

    for index, k in enumerate(mesh):
        if completed[index]:
            continue
        guard_resources(f"neutrality:k{index}:before")
        result = model.solve(k, sigma, states)
        guard_resources(f"neutrality:k{index}:after")
        first = int(result["first_index"])
        energies = np.asarray(result["energies"])
        top = occupied - 1 - first
        bottom = occupied - first
        if not (0 <= top and bottom < len(energies)):
            raise SystemExit(
                f"Window of {states} states does not bracket the neutrality index at "
                f"k={k}; increase --states."
            )
        eigenvalues_array[index] = energies
        first_indices_array[index] = first
        completed[index] = True
        atomic_savez(
            progress_path,
            eigenvalues=eigenvalues_array,
            first_indices=first_indices_array,
            completed=completed,
            signature=np.asarray(progress_signature),
        )

    homo = []
    lumo = []
    for energies, first in zip(eigenvalues_array, first_indices_array):
        homo.append(float(energies[occupied - 1 - first]))
        lumo.append(float(energies[occupied - first]))

    homo_edge, lumo_edge = max(homo), min(lumo)
    gapped = lumo_edge > homo_edge
    if gapped:
        fermi = 0.5 * (homo_edge + lumo_edge)
        method = "uniform_kmesh_orthogonal_inertia_midgap"
    else:
        # The manifold is gapless on this mesh (Dirac points survive at K), so the
        # mid-gap formula is undefined. E_F is bracketed by bisecting the integrated
        # state count: the level where strictly-below occupancy first reaches n_occ*n_k.
        # The count is a step function, so bisection converges onto an eigenvalue. The
        # convention is then made explicit below.
        target = occupied * len(mesh)
        counted = lambda level: sum(
            base + int((window < level).sum())
            for base, window in zip(first_indices_array, eigenvalues_array)
        )
        low, high = lumo_edge, homo_edge
        for _ in range(100):
            middle = 0.5 * (low + high)
            if counted(middle) < target:
                low = middle
            else:
                high = middle
        # `high` is the first level whose strictly-below count reaches the target, i.e.
        # the lowest unoccupied level. Publish the midpoint between the adjacent
        # occupied and unoccupied levels so that E_F is not pinned onto an eigenvalue
        # and `states strictly below E_F` is unambiguously n_occ*n_k.
        flat = np.sort(eigenvalues_array.ravel())
        below_high = flat[flat < high]
        occupied_level = float(below_high[-1]) if below_high.size else float(low)
        fermi = 0.5 * (occupied_level + float(high))
        method = "uniform_kmesh_orthogonal_inertia_state_counting_midpoint"

    cache_signature = mesh_cache_signature(geometry, mesh_size, states, sigma, occupied)
    atomic_savez(
        RESULTS / "stages" / MESH_CACHE,
        eigenvalues=eigenvalues_array,
        first_indices=first_indices_array,
        signature=np.asarray(cache_signature),
    )
    smeared = {
        f"{1000 * temperature:.1f}_meV": smeared_chemical_potential(
            eigenvalues_array, first_indices_array, occupied, temperature
        )
        for temperature in (0.0005, 0.001)
    }
    maximum_smearing_change = max(abs(value - fermi) for value in smeared.values())
    return {
        "stage": "neutrality",
        "chemical_potential_available": True,
        "energy_eV": float(fermi),
        # Positive => a real global gap. Negative => valence and conduction overlap
        # across the mesh, i.e. the manifold is gapless; it is NOT a gap.
        "finite_mesh_gap_eV": float(lumo_edge - homo_edge),
        "gapped": bool(gapped),
        "homo_edge_eV": float(homo_edge),
        "kpoint_count": int(len(mesh)),
        "lumo_edge_eV": float(lumo_edge),
        "matrix_dimension": int(model.count),
        "mesh": [mesh_size, mesh_size, 1],
        "method": method,
        "fermi_convention": (
            "midpoint between the highest level with strictly-below count < n_occ*n_k "
            "and the lowest level reaching it; states strictly below E_F equal n_occ*n_k"
        ),
        "smearing_diagnostic": {
            "chemical_potential_eV": smeared,
            "maximum_change_from_zero_temperature_eV": maximum_smearing_change,
            "stable_within_1meV": bool(maximum_smearing_change < 0.001),
        },
        "neutral_electrons": int(model.count),
        "scope": "Moon-Koshino p_z tight binding; approximate reference, not a DFT ground truth",
        "sigma_eV": float(sigma),
        "spin_degeneracy": SPIN_DEGENERACY,
        "states_per_k": int(states),
        "status": "computed",
        "target_occupied_bands_per_k": int(occupied),
        "mesh_cache_signature": cache_signature,
        "mesh_cache_sha256": file_sha256(RESULTS / "stages" / MESH_CACHE),
        "wall_seconds": time.time() - started,
    }


def stage_bands(states: int = BAND_STATES) -> dict:
    started = time.time()
    guard_resources("bands:start")
    geometry = read_geometry(TARGET_FDF)
    validate_target_geometry(geometry)
    model = PzTightBinding(geometry)
    layers = layer_labels(model.xyz)
    lower = layers == 0
    neutrality = read_stage("neutrality")
    sigma = neutrality.get("energy_eV")
    if sigma is None:
        sigma = read_stage("preflight").get("sigma_eV") or dirac_level()["dirac_energy_eV"]
    fermi = neutrality.get("energy_eV", 0.0)
    occupied = model.count // 2
    path = band_path(model.cell)

    progress_path = RESULTS / "stages" / "bands_progress.json"
    progress = {}
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = {}
    expected_progress_signature = payload_sha256(
        {
            "geometry_sha256": geometry_fingerprint(geometry),
            "model_sha256": payload_sha256(MODEL_PROVENANCE),
            "states": states,
            "k_path": path["contract"],
            "sigma_eV": float(sigma),
        }
    )
    if progress.get("signature") != expected_progress_signature:
        progress = {"signature": expected_progress_signature, "points": {}}
    point_results = progress.setdefault("points", {})

    for index, k in enumerate(path["fractional"]):
        if str(index) in point_results:
            continue
        guard_resources(f"bands:k{index}:before")
        result = model.solve(k, sigma, states, vectors=True)
        guard_resources(f"bands:k{index}:after")
        weights = np.abs(result["states"]) ** 2
        weights /= weights.sum(axis=0)
        lower_weight = weights[lower].sum(axis=0)
        point_rows = []
        for rank, energy in enumerate(result["energies"]):
            absolute_index = int(result["first_index"] + rank)
            point_rows.append(
                {
                    "absolute_band_index": absolute_index,
                    "band_index": absolute_index,
                    "solver_band_index": rank,
                    "energy_aligned_eV": float(energy - fermi),
                    "energy_eV": float(energy),
                    "flat_manifold": bool(
                        occupied - 2 <= absolute_index <= occupied + 1
                    ),
                    "k_distance": float(path["distance"][index]),
                    "k_index": index,
                    "k_label": path["labels"][index],
                    "weight_c_pz": 1.0,
                    "weight_c_total": 1.0,
                    "weight_graphene_lower": float(lower_weight[rank]),
                    "weight_graphene_upper": float(1.0 - lower_weight[rank]),
                    "weight_hbn": 0.0,
                }
            )
        point_results[str(index)] = {
            "residual_eV": float(result["residual_eV"]),
            "rows": point_rows,
        }
        atomic_write_json(progress_path, progress)

    rows = []
    residuals = []
    for index in range(len(path["fractional"])):
        point = point_results.get(str(index))
        if not point:
            raise ResourceGuardError(f"bands: missing resumable point {index}")
        rows.extend(point["rows"])
        residuals.append(float(point["residual_eV"]))
    return {
        "stage": "bands",
        "status": "completed",
        "bands": rows,
        "k_path": path["contract"],
        "max_residual_eV": float(np.max(residuals)),
        "num_bands": int(states),
        "occupied_bands_per_k": int(occupied),
        "sigma_eV": float(sigma),
        "wall_seconds": time.time() - started,
    }


def partial_dos_from_cache(
    eigenvalues: np.ndarray,
    fermi: float,
    matrix_dimension: int,
    *,
    broadening_eV: float = DOS_BROADENING_EV,
    requested_window_eV: float = DOS_WINDOW_EV,
) -> dict:
    lower_coverage = fermi - np.min(eigenvalues, axis=1)
    upper_coverage = np.max(eigenvalues, axis=1) - fermi
    fully_covered = float(min(np.min(lower_coverage), np.min(upper_coverage)))
    valid_half_width = min(
        requested_window_eV, max(0.0, fully_covered - 5 * broadening_eV)
    )
    if valid_half_width <= 0:
        raise RuntimeError("Cached eigenvalue window is too narrow for a valid broadened DOS")
    points = max(101, int(round(600 * valid_half_width / requested_window_eV)) + 1)
    grid = np.linspace(fermi - valid_half_width, fermi + valid_half_width, points)
    flat = eigenvalues.ravel()
    prefactor = SPIN_DEGENERACY / (
        len(eigenvalues) * broadening_eV * math.sqrt(2.0 * math.pi)
    )
    density = prefactor * np.exp(
        -0.5 * ((grid[:, None] - flat[None, :]) / broadening_eV) ** 2
    ).sum(axis=1)
    return {
        "broadening_eV": broadening_eV,
        "fully_covered_half_width_eV": fully_covered,
        "published_half_width_eV": valid_half_width,
        "coverage_margin_eV": 5 * broadening_eV,
        "kpoint_count": int(len(eigenvalues)),
        "normalization": "states_per_eV_per_moire_cell_including_spin_degeneracy_2",
        "low_energy_dos": [
            {
                "dos": float(value),
                "dos_per_atom": float(value / matrix_dimension),
                "energy_aligned_eV": float(energy - fermi),
                "energy_eV": float(energy),
            }
            for energy, value in zip(grid, density)
        ],
        "scope": "partial DOS only inside the fully covered shift-invert window",
        "states_per_k": int(eigenvalues.shape[1]),
    }


def stage_dos() -> dict:
    """Gaussian-broadened DOS of the low-energy window from the neutrality mesh."""
    neutrality = read_stage("neutrality")
    if not neutrality:
        raise SystemExit("Run --neutrality first: the DOS reuses its mesh eigenvalues.")
    cache = load_mesh_cache(neutrality.get("mesh_cache_signature"))
    result = partial_dos_from_cache(
        cache["eigenvalues"],
        float(neutrality["energy_eV"]),
        int(neutrality["matrix_dimension"]),
    )
    return {
        "stage": "dos",
        "status": "completed",
        "mesh": neutrality.get("mesh"),
        "fermi_level_eV": float(neutrality["energy_eV"]),
        "mesh_cache_signature": cache["signature"],
        "mesh_cache_sha256": cache["sha256"],
        **result,
    }


def path_manifold_observables(bands: list[dict], occupied: int) -> dict:
    manifold = [row for row in bands if row["flat_manifold"]]
    if not manifold:
        return {"status": "unavailable"}
    energies = np.array([row["energy_aligned_eV"] for row in manifold])
    indices = np.array([row["absolute_band_index"] for row in manifold])
    valence = energies[indices <= occupied - 1]
    conduction = energies[indices >= occupied]
    remote_above = [
        row["energy_aligned_eV"] for row in bands if row["absolute_band_index"] == occupied + 2
    ]
    remote_below = [
        row["energy_aligned_eV"] for row in bands if row["absolute_band_index"] == occupied - 3
    ]
    # Valence and conduction are split by absolute band index, never by the sign of
    # E - E_F: at K the four states are degenerate and a sign rule would misclassify them.
    # A negative value below is an energy overlap, not a gap.
    remote_conduction = float(min(remote_above) - max(conduction)) if remote_above else None
    remote_valence = float(min(valence) - max(remote_below)) if remote_below else None
    neutrality_gap = float(min(conduction) - max(valence))
    return {
        "scope": "sampled_k_gamma_m_k_path_only_not_full_brillouin_zone",
        "band_count": 4,
        "path_remote_conduction_gap_eV": remote_conduction,
        "path_remote_valence_gap_eV": remote_valence,
        "path_isolated_from_remote_bands": bool(
            (remote_conduction or 0) > 0 and (remote_valence or 0) > 0
        ),
        "path_manifold_width_eV": float(energies.max() - energies.min()),
        "path_neutrality_gap_eV": neutrality_gap,
        "path_neutrality_gapless": bool(neutrality_gap <= 0),
        "note": "negative values are band overlaps, not gaps; rigid unrelaxed geometry",
        "selection": "fixed band indices n_occ-2..n_occ+1, not the flattest branches",
        "status": "computed",
    }


def stage_aggregate() -> dict:
    preflight = read_stage("preflight")
    neutrality = read_stage("neutrality")
    bands = read_stage("bands")
    dos = read_stage("dos")
    if not bands:
        raise SystemExit("Nothing to aggregate: run --bands first.")
    # Aggregation is strictly read-only upstream. Re-stamping the stage files with the
    # current script SHA would erase the distinction between the code that COMPUTED a
    # result and the code that merely collected it, which is exactly the provenance the
    # summary is supposed to certify.
    stages = {
        "preflight": preflight,
        "neutrality": neutrality,
        "bands": bands,
        "dos": dos,
    }
    provenance = {
        "aggregation_implementation": implementation_identity(),
        "aggregate_rewrites_upstream_stages": False,
        "source_implementation": {
            name: (payload.get("implementation") or "not_recorded")
            for name, payload in stages.items()
            if payload
        },
        "source_input_contract_sha256": {
            name: payload.get("input_contract", {}).get("contract_sha256", "not_recorded")
            for name, payload in stages.items()
            if payload
        },
        "missing_stages": [name for name, payload in stages.items() if not payload],
    }
    # Resource monitoring is reported per stage. A stage that predates the runtime guard
    # reports "unknown"; the summary must not invent a global maximum out of whichever
    # stages happen to carry numbers, least of all out of the aggregation process itself.
    per_stage_resources = {
        name: (payload.get("resource_observations") or "not_recorded")
        for name, payload in stages.items()
        if payload
    }
    recorded_temperatures = [
        value.get("maximum_cpu_temperature_c")
        for value in per_stage_resources.values()
        if isinstance(value, dict) and value.get("maximum_cpu_temperature_c") is not None
    ]
    resources = {
        "per_stage": per_stage_resources,
        "complete": len(recorded_temperatures) == len(per_stage_resources),
        "maximum_cpu_temperature_c": (
            max(recorded_temperatures)
            if recorded_temperatures and len(recorded_temperatures) == len(per_stage_resources)
            else None
        ),
        "scope": "per-stage observations; a global maximum is published only when every stage recorded one",
    }

    cache_available = (RESULTS / "stages" / MESH_CACHE).exists() and bool(neutrality)
    if cache_available:
        cache = load_mesh_cache(neutrality.get("mesh_cache_signature"))
        mesh_observables = mesh_observables_from_cache(
            cache["eigenvalues"],
            cache["first_indices"],
            int(bands["occupied_bands_per_k"]),
            list(neutrality.get("mesh") or [NEUTRALITY_MESH, NEUTRALITY_MESH, 1]),
        )
        mesh_observables["mesh_cache_sha256"] = cache["sha256"]
    else:
        mesh_observables = {
            "status": "unavailable",
            "reason": "neutrality mesh cache missing",
        }
    normalized_bands = [
        {
            **point,
            "solver_band_index": point.get("solver_band_index", point.get("band_index")),
            "band_index": point.get("absolute_band_index", point.get("band_index")),
        }
        for point in bands.get("bands", [])
    ]
    bands = {**bands, "bands": normalized_bands}
    cell = np.asarray(bands["k_path"]["direct_lattice_vectors_ang"])
    path_observables = path_manifold_observables(
        bands["bands"], bands["occupied_bands_per_k"]
    )
    row = {
        "backend_effective": "cpu_scipy_shift_invert",
        "band_ordering": {
            "band_character_continuity_claimed": False,
            "method": "ascending_energy_rank_per_k",
        },
        "bands": bands["bands"],
        "comparison_limitations": {
            "graph2mat_absolute_band_indices_available": False,
            "graph2mat_global_manifold_comparison_available": False,
            "permitted_graph2mat_comparisons": [
                "energy-referenced visual overlay",
                "labelled high-symmetry splittings at K/Gamma/M",
            ],
            "reason": "existing Graph2Mat path artifacts do not persist per-k absolute band indices",
        },
        "fermi_level_eV": neutrality.get("energy_eV"),
        "flat_manifold": path_observables,
        "path_observables": path_observables,
        "mesh_observables": mesh_observables,
        "k_path": bands["k_path"],
        "low_energy_dos": dos.get("low_energy_dos", []),
        "material_system": "pure_tbg",
        "max_residual_eV": bands.get("max_residual_eV"),
        "model": "tight_binding",
        "provenance": provenance,
        "model_provenance": MODEL_PROVENANCE,
        "moire_cell_area_ang2": float(abs(np.cross(cell[0], cell[1])[2])),
        "neutrality_reference": neutrality,
        "num_bands": bands.get("num_bands"),
        "occupied_bands_per_k": bands["occupied_bands_per_k"],
        "preflight": preflight,
        "projection": {
            "scientific_status": "exact_for_an_orthogonal_pz_basis",
            "status": "completed",
        },
        "scientific_status": "independent_approximate_reference_not_dft_ground_truth",
        "seed": None,
        "resource_observations": resources,
        "solver_threads": SOLVER_THREADS,
        "status": "completed",
        "training_size": None,
        "twist_angle_deg": 1.084549049,
        "visible_band_tier": "tight_binding_moon_koshino",
        "warning": "TB atomístico pz: referencia aproximada, no ground truth DFT.",
    }
    summary = {
        "campaign_kind": "pure_tbg_atomistic_pz_tight_binding",
        "provenance": provenance,
        "model_provenance": MODEL_PROVENANCE,
        "scientific_status": "completed",
        "spectra": [row],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    guard_disk()
    path = RESULTS / "summary" / "spectral_results.json"
    atomic_write_json(path, summary)
    return {"stage": "aggregate", "path": str(path), "bands": len(bands["bands"])}


# --------------------------------------------------------------------------------------
# self test


def self_test() -> None:
    """Physics and numerics on cells small enough for a dense reference."""
    monolayer = monolayer_geometry()
    assert len(monolayer) == 2, len(monolayer)
    dirac = dirac_level(monolayer)
    # The FDF coordinates are rounded at ~1e-9 Ang, which breaks the exact C3 symmetry
    # and splits the Dirac point by dh/dd * 1e-9 ~ 1e-8 eV. That is the geometric floor
    # on any degeneracy claim, not a solver error; the coordinates are used verbatim.
    assert dirac["degeneracy_splitting_eV"] < 1e-7, dirac
    assert 5e5 < dirac["v_fermi_m_per_s"] < 1.2e6, dirac
    # Second and third neighbours break particle-hole symmetry: E_D must not be pinned to 0.
    assert abs(dirac["dirac_energy_eV"]) > 1e-3, dirac

    first_neighbour = hopping(np.array([1.4318286667]), np.array([0.0]))[0]
    assert -2.64 < first_neighbour < -2.62, first_neighbour
    vertical = hopping(np.array([3.35]), np.array([3.35]))[0]
    assert abs(vertical - V_PP_SIGMA_0) < 1e-12, vertical

    bilayer_aa = PzTightBinding(read_geometry(REPO / "materials/bilayer_graphene_AA/RUN.fdf"))
    k_corner = np.array([*corner_k(bilayer_aa.cell), 0.0])
    for k in ([0.0, 0.0, 0.0], k_corner, [0.17, 0.41, 0.0]):
        assert bilayer_aa.hermiticity_error(k) < 1e-12, k
    # AA stacking: two rigidly split copies of the monolayer cone at K.
    aa_at_k = np.linalg.eigvalsh(bilayer_aa.hk(k_corner).toarray())
    assert abs((aa_at_k[1] - aa_at_k[0]) - (aa_at_k[3] - aa_at_k[2])) < 1e-9, aa_at_k

    ab = PzTightBinding(read_geometry(REPO / "materials/bilayer_graphene_AB/RUN.fdf"))
    ba = PzTightBinding(read_geometry(REPO / "materials/bilayer_graphene_BA/RUN.fdf"))
    for k in ([0.11, 0.37, 0.0], k_corner):
        left = np.linalg.eigvalsh(ab.hk(k).toarray())
        right = np.linalg.eigvalsh(ba.hk(k).toarray())
        assert np.max(np.abs(left - right)) < 1e-9, (k, left, right)
        # Time reversal.
        reversed_k = np.linalg.eigvalsh(ab.hk([-value for value in k]).toarray())
        assert np.max(np.abs(left - reversed_k)) < 1e-9, k

    # Global translation (including one that crosses the cell boundary) and atom
    # permutation must leave the spectrum untouched.
    base = read_geometry(REPO / "materials/bilayer_graphene_AB/RUN.fdf")
    reference = np.linalg.eigvalsh(ab.hk([0.11, 0.37, 0.0]).toarray())
    moved = base.copy()
    moved.xyz[:] = base.xyz + np.array([0.83, -1.27, 0.0]) + base.cell[0]
    shifted = np.linalg.eigvalsh(
        PzTightBinding(moved).hk([0.11, 0.37, 0.0]).toarray()
    )
    assert np.max(np.abs(reference - shifted)) < 1e-10, shifted - reference
    order = [3, 1, 0, 2]
    permuted = np.linalg.eigvalsh(
        PzTightBinding(base.sub(order)).hk([0.11, 0.37, 0.0]).toarray()
    )
    assert np.max(np.abs(reference - permuted)) < 1e-10, permuted - reference

    # Sparse shift-invert window and the inertia count against a dense reference.
    supercell = PzTightBinding(base.tile(14, 0).tile(14, 1))
    for k in ([0.11, 0.37, 0.0], [0.0, 0.0, 0.0]):
        dense = np.linalg.eigvalsh(supercell.hk(k).toarray())
        for sigma in (-0.37, 0.05, 1.31):
            result = supercell.solve(k, sigma, 20, vectors=True)
            assert result["states_below_sigma"] == int((dense < sigma).sum()), (
                k, sigma, result["states_below_sigma"], int((dense < sigma).sum())
            )
            window = dense[result["first_index"]: result["first_index"] + 20]
            assert np.max(np.abs(window - result["energies"])) < 1e-9, (k, sigma)
            assert result["residual_eV"] < 1e-7, result["residual_eV"]

    print("self-test OK")


def contract_for_stage(name: str, args: argparse.Namespace) -> dict:
    if name == "preflight":
        return stage_contract(
            name,
            band_states=BAND_STATES,
            cutoff_sensitivity=[4.0, 5.0],
            geometry_scaled_a_reference_ang=2.48,
            sigma_offsets_eV=[-0.05, 0.05],
            validation_version=2,
        )
    if name == "neutrality":
        # sigma comes from preflight; without it in the contract a changed preflight
        # would leave a stale neutrality looking reusable.
        preflight = read_stage("preflight")
        return stage_contract(
            name,
            mesh=args.mesh,
            states=args.states,
            sigma_eV=preflight.get("sigma_eV"),
        )
    if name == "bands":
        neutrality = read_stage("neutrality")
        return stage_contract(
            name,
            states=args.band_states,
            points_per_segment=11,
            sigma_eV=float(neutrality.get("energy_eV")) if neutrality else None,
        )
    if name == "dos":
        # The DOS consumes E_F and the spectral cache. Both must bind the contract, or a
        # recomputed neutrality leaves a DOS recentred on a stale E_F looking reusable.
        neutrality = read_stage("neutrality")
        return stage_contract(
            name,
            broadening_eV=DOS_BROADENING_EV,
            requested_window_eV=DOS_WINDOW_EV,
            mesh=neutrality.get("mesh"),
            states=neutrality.get("states_per_k"),
            coverage_margin_sigma=5,
            fermi_level_eV=neutrality.get("energy_eV"),
            neutrality_sha256=payload_sha256(neutrality) if neutrality else None,
            mesh_cache_sha256=file_sha256(RESULTS / "stages" / MESH_CACHE),
        )
    return stage_contract(
        name,
        preflight_sha256=payload_sha256(read_stage("preflight")),
        neutrality_sha256=payload_sha256(read_stage("neutrality")),
        bands_sha256=payload_sha256(read_stage("bands")),
        dos_sha256=payload_sha256(read_stage("dos")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--neutrality", action="store_true")
    parser.add_argument("--bands", action="store_true")
    parser.add_argument("--dos", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mesh", type=int, default=NEUTRALITY_MESH)
    parser.add_argument("--states", type=int, default=MESH_STATES)
    parser.add_argument("--band-states", type=int, default=BAND_STATES)
    parser.add_argument(
        "--force", action="store_true", help="recompute selected stages even if contracts match"
    )
    args = parser.parse_args()

    if args.self_test:
        guard_resources("self-test:start")
        self_test()
        return
    if args.all:
        guard_resources("self-test:start")
        self_test()
    stages = []
    if args.all or args.preflight:
        stages.append(("preflight", lambda: stage_preflight()))
    if args.all or args.neutrality:
        stages.append(("neutrality", lambda: stage_neutrality(args.mesh, args.states)))
    if args.all or args.bands:
        stages.append(("bands", lambda: stage_bands(args.band_states)))
    if args.all or args.dos:
        stages.append(("dos", stage_dos))
    if args.all or args.aggregate:
        stages.append(("aggregate", stage_aggregate))
    if not stages:
        parser.error("choose at least one stage (or --all)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        RESULTS / "status.json",
        {
            "status": "running",
            "selected_stages": [name for name, _ in stages],
            "resource_observations": dict(RESOURCE_OBSERVATIONS),
        },
    )
    try:
        for name, runner in stages:
            contract = contract_for_stage(name, args)
            if not args.force and stage_reusable(name, contract):
                print(f"[{name}] reused: input contract matches", flush=True)
                continue
            started = time.time()
            guard_resources(f"{name}:start")
            payload = with_runtime_metadata(runner(), contract)
            payload.setdefault("status", "completed")
            write_stage(name, payload)
            summary = {
                key: value
                for key, value in payload.items()
                if not isinstance(value, (list, dict))
            }
            print(
                f"[{name}] {time.time() - started:.1f}s "
                f"{json.dumps(summary, sort_keys=True)}",
                flush=True,
            )
        RESOURCE_OBSERVATIONS["final_cpu_temperature_c"] = cpu_package_temperature_c()
        atomic_write_json(
            RESULTS / "status.json",
            {
                "status": "completed",
                "selected_stages": [name for name, _ in stages],
                "resource_observations": dict(RESOURCE_OBSERVATIONS),
            },
        )
    except BaseException as error:
        RESOURCE_OBSERVATIONS["final_cpu_temperature_c"] = cpu_package_temperature_c()
        atomic_write_json(
            RESULTS / "status.json",
            {
                "status": "stopped_safely" if isinstance(error, ResourceGuardError) else "failed",
                "reason": f"{type(error).__name__}: {error}",
                "selected_stages": [name for name, _ in stages],
                "resource_observations": dict(RESOURCE_OBSERVATIONS),
                "resumable": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
