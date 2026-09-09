#!/usr/bin/env python3
"""D01/D03 / E-F_001-S30: the commensurate ``q != 0`` reference (route Q-A).

Graphene ``K`` is the first EPC gate that ``Gamma`` cannot see, and this module
is the *reference* answer to it — the transparent one, not the scalable one. A
commensurate supercell turns a finite-``q`` perturbation into a cell-periodic
one, so every certified piece of the ``Gamma`` chain is reused verbatim:

    q, M  ->  supercell in which exp(2i pi q.R) is periodic   (CommensurateCell)
    e_qnu ->  u_l = u(q) exp(2i pi q.R_l), split Re / Im       (ComplexMode)
    cos, sin  ->  two ordinary Gamma D_H + basis responses      (existing chain)
    D_H(q) = a_cos D_H[cos] + i a_sin D_H[sin]                  (ComplexModeResponse)

Three things this file refuses to blur:

* **The artifact ``q`` is the supercell's, and it is Gamma.** In route Q-A the
  finite ``q`` lives in the displacement pattern, not in the Bloch phases of the
  blocks, so :class:`ComplexModeResponse` requires both halves to be signed at
  ``q = 0`` and records the *primitive* ``q`` separately. Relabelling a Gamma
  artifact ``q = K`` is the substitution GO-8b forbids; producing the same
  number through a bigger cell is the reference GO-8b is validated *against*.
* **``cos`` and ``sin`` stay separate artifacts.** They are two independent,
  separately signed, real directional responses. The complex mode is their
  linear combination, taken here, with the two Frobenius amplitudes that the
  unit-norm :class:`~fd_perturbation_space.Direction` normalisation removed.
  (F1)/(F2) are linear in the response, which is what makes this exact rather
  than approximate.
* **``k -> k+q`` is a subspace statement, not a band index.** ``k`` and ``k+q``
  fold onto the *same* supercell ``k`` (that is what commensurate means), so
  both live in one eigenspace and are told apart by their primitive Bloch
  character, measured in the ``S`` metric with the GO-6 machinery. No band
  ordering, no sorting, no eigenvalue matching.

Everything produced here is ``candidate__pending_go5``: GO-5 is what promotes
it, and nothing in this module may claim to be it.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from artifact_signature import input_signature_sha256  # noqa: E402
from compute_epc_matrix_elements import (  # noqa: E402
    Eigenspace,
    EpcContractionError,
    PaoCovariantResponse,
)
from epc_formalism import (  # noqa: E402
    REPRESENTATION_D_S,
    REPRESENTATION_S_L_S_R,
    BasisResponse,
    EpcBlock,
    basis_response_D_S,
    basis_response_S_L_S_R,
    epc_matrix_elements,
    gauge_invariant_block_metrics,
    pao_covariant_response,
)
from epc_fourier import CONVENTION_ID, k_plus_q, require_same_conventions  # noqa: E402
from epc_subspaces import (  # noqa: E402
    block_metrics,
    eigenvalue_resolutions_ev,
    near_degenerate_clusters,
    subspace_metrics,
    subspace_overlap,
)
from fd_perturbation_space import collective  # noqa: E402
from phonon_provider import PhononMode  # noqa: E402

SCHEMA = "epc_commensurate_q_reference_v1"
TICKET = "D01/D03 / E-F_001-S30"

#: Route Q-A of the roadmap: the finite q is carried by a bigger cell, not by a
#: phase applied to Gamma blocks. Q-B/Q-C are validated against this.
Q_REPRESENTATION = "commensurate_supercell_cos_sin"

#: Nothing this module builds is validated physics until GO-5 says so.
CANDIDATE_STATUS = "candidate__pending_go5"

#: A commensurability residual is a rational with denominator |det M|; anything
#: this side of it is float noise, anything beyond it is a wrong supercell.
COMMENSURABILITY_TOL = 1e-9

#: Chained products against S on the projection path (projector Gram, Loewdin,
#: Ritz, C†SC), the same accounting compute_epc_matrix_elements uses for its
#: dense Loewdin window: same policy, different arithmetic.
PROJECTION_PRODUCTS = 4

#: Below this, Im(u) is not a direction: it cannot be normalised, and a mode
#: whose sin part vanishes is real and does not need this module.
AMPLITUDE_TOL_ANG = 1e-12


class CommensurateError(RuntimeError):
    """A commensurate q-reference was requested that the geometry cannot carry."""


# --------------------------------------------------------------------------- #
# The cell that makes q periodic
# --------------------------------------------------------------------------- #


def _integer_matrix(matrix: Any) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3):
        raise CommensurateError(f"the supercell matrix must be (3, 3), got {values.shape}")
    rounded = np.rint(values)
    if np.abs(values - rounded).max() > 1e-9:
        raise CommensurateError(f"the supercell matrix must be integer, got {values.tolist()}")
    if abs(np.linalg.det(rounded)) < 0.5:
        raise CommensurateError("the supercell matrix is singular")
    return rounded.astype(np.int64)


def diagonal_supercell_matrix(q: Any, *, max_denominator: int = 64) -> np.ndarray:
    """The dumbest matrix that works: ``diag`` of the denominators of ``q``.

    Always valid, never minimal — the ``sqrt(3) x sqrt(3)`` cell of graphene
    ``K`` is three cells and this is nine. Pass the minimal matrix explicitly
    when the point group hands you one; the constructor validates it either way.
    """
    from fractions import Fraction

    diagonal = []
    for value in np.asarray(q, dtype=np.float64).reshape(3):
        fraction = Fraction(float(value)).limit_denominator(int(max_denominator))
        if abs(float(fraction) - float(value)) > COMMENSURABILITY_TOL:
            raise CommensurateError(
                f"q component {value!r} is not a rational with denominator <= {max_denominator}: "
                "no finite commensurate supercell exists for it"
            )
        diagonal.append(fraction.denominator)
    return np.diag(diagonal).astype(np.int64)


def _representative_images(matrix: np.ndarray) -> np.ndarray:
    """The ``|det M|`` primitive lattice points inside one supercell.

    ``l`` is a primitive-lattice integer triple; its supercell-fractional
    coordinate is ``f = l @ M^-1``, and the representatives are ``f`` in
    ``[0, 1)``. Ordered by ``f``, so the home cell ``(0, 0, 0)`` is first.
    """
    inverse = np.linalg.inv(matrix.astype(np.float64))
    expected = int(round(abs(np.linalg.det(matrix.astype(np.float64)))))
    bound = int(np.abs(matrix).sum(axis=0).max())
    found: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
    for image in itertools.product(range(-bound, bound + 1), repeat=3):
        fractional = np.asarray(image, dtype=np.float64) @ inverse
        if np.all(fractional > -COMMENSURABILITY_TOL) and np.all(
            fractional < 1.0 - COMMENSURABILITY_TOL
        ):
            found.append((tuple(np.round(fractional, 9)), tuple(int(v) for v in image)))
    if len(found) != expected:
        raise CommensurateError(
            f"found {len(found)} lattice representatives for a supercell of |det M| = {expected}; "
            "the search box or the matrix is wrong"
        )
    found.sort()
    return np.array([image for _, image in found], dtype=np.int64)


@dataclass(frozen=True)
class CommensurateCell:
    """A supercell in which ``exp(2i pi q.R)`` is periodic, and its atom map.

    ``matrix`` rows are the supercell lattice vectors in primitive lattice
    units, so ``cell_sc = M @ cell_prim`` and ``k_frac_sc = M @ k_frac_prim``.
    Commensurability of ``q`` is exactly ``M @ q`` integer.
    """

    q_primitive: tuple[float, float, float]
    matrix: np.ndarray
    primitive_cell_ang: np.ndarray
    primitive_positions_ang: np.ndarray
    images: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        matrix = _integer_matrix(self.matrix)
        q = np.asarray(self.q_primitive, dtype=np.float64).reshape(3)
        residual = matrix.astype(np.float64) @ q
        if np.abs(residual - np.rint(residual)).max() > COMMENSURABILITY_TOL:
            raise CommensurateError(
                f"q = {q.tolist()} is not commensurate with the supercell: M @ q = "
                f"{residual.tolist()} is not integer, so exp(2i pi q.R) is not periodic in it"
            )
        cell = np.asarray(self.primitive_cell_ang, dtype=np.float64)
        positions = np.asarray(self.primitive_positions_ang, dtype=np.float64)
        if cell.shape != (3, 3):
            raise CommensurateError(f"primitive_cell_ang must be (3, 3), got {cell.shape}")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise CommensurateError(
                f"primitive_positions_ang must be (na, 3), got {positions.shape}"
            )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "q_primitive", tuple(float(value) for value in q))
        object.__setattr__(self, "primitive_cell_ang", cell)
        object.__setattr__(self, "primitive_positions_ang", positions)
        object.__setattr__(
            self,
            "images",
            _representative_images(matrix) if self.images is None else np.asarray(self.images),
        )

    # -- geometry ------------------------------------------------------------
    @property
    def multiplicity(self) -> int:
        """``N_c``: how many primitive cells the supercell holds."""
        return int(self.images.shape[0])

    @property
    def primitive_atom_count(self) -> int:
        return int(self.primitive_positions_ang.shape[0])

    @property
    def atom_count(self) -> int:
        return self.multiplicity * self.primitive_atom_count

    @property
    def cell_ang(self) -> np.ndarray:
        return self.matrix.astype(np.float64) @ self.primitive_cell_ang

    @property
    def positions_ang(self) -> np.ndarray:
        """Atom-major by image: atom ``i`` is ``(image i // na, kappa i % na)``."""
        offsets = self.images.astype(np.float64) @ self.primitive_cell_ang
        return np.concatenate([self.primitive_positions_ang + row for row in offsets])

    @property
    def atom_image_index(self) -> np.ndarray:
        return np.repeat(np.arange(self.multiplicity), self.primitive_atom_count)

    def fold(self, k_primitive: Any) -> np.ndarray:
        """``k`` of the primitive cell in supercell fractional coordinates."""
        folded = self.matrix.astype(np.float64) @ np.asarray(
            k_primitive, dtype=np.float64
        ).reshape(3)
        return folded - np.rint(folded)

    def folds_together(self, k_a: Any, k_b: Any) -> bool:
        """Do two primitive ``k`` land on the same supercell ``k``?"""
        difference = self.matrix.astype(np.float64) @ (
            np.asarray(k_a, dtype=np.float64).reshape(3)
            - np.asarray(k_b, dtype=np.float64).reshape(3)
        )
        return bool(np.abs(difference - np.rint(difference)).max() <= COMMENSURABILITY_TOL)

    def orbital_image_index(self, orbitals_per_primitive_atom: Sequence[int]) -> np.ndarray:
        """Image index of every supercell orbital, in the atom-major ordering."""
        counts = np.asarray(orbitals_per_primitive_atom, dtype=np.int64).reshape(-1)
        if counts.size != self.primitive_atom_count:
            raise CommensurateError(
                f"{counts.size} orbital counts for {self.primitive_atom_count} primitive atoms"
            )
        per_cell = int(counts.sum())
        return np.repeat(np.arange(self.multiplicity), per_cell)

    # -- provenance ----------------------------------------------------------
    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "ticket": TICKET,
            "q_representation": Q_REPRESENTATION,
            "convention_id": CONVENTION_ID,
            "q_primitive": list(self.q_primitive),
            "q_supercell": self.fold(self.q_primitive).tolist(),
            "supercell_matrix": self.matrix.tolist(),
            "multiplicity": self.multiplicity,
            "primitive_atom_count": self.primitive_atom_count,
            "atom_count": self.atom_count,
            "images": self.images.tolist(),
            "supercell_hash": self.supercell_hash,
        }

    @property
    def supercell_hash(self) -> str:
        return input_signature_sha256(
            {
                "schema": SCHEMA,
                "q_primitive": [float(value) for value in self.q_primitive],
                "supercell_matrix": self.matrix.tolist(),
                "primitive_cell_ang": self.primitive_cell_ang.tolist(),
                "primitive_positions_ang": self.primitive_positions_ang.tolist(),
                "images": self.images.tolist(),
            }
        )


# --------------------------------------------------------------------------- #
# The mode: one complex pattern, two real directions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComplexMode:
    """``u_l = u(q) exp(2i pi q.R_l)`` on the supercell, split into Re and Im.

    ``cos``/``sin`` are ordinary unit-Frobenius directions — the only kind the
    derivative backends accept — and the two amplitudes in Ang are what puts
    the physical displacement back together:
    ``u = a_cos * cos + i * a_sin * sin``.
    """

    cell: CommensurateCell
    cos: Any  # fd_perturbation_space.Direction
    sin: Any
    cos_amplitude_ang: float
    sin_amplitude_ang: float
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def pattern_ang(self) -> np.ndarray:
        """The complex displacement, reassembled from the two signed halves."""
        return (
            self.cos_amplitude_ang * self.cos.vectors
            + 1j * self.sin_amplitude_ang * self.sin.vectors
        )

    @property
    def complex_mode_hash(self) -> str:
        return input_signature_sha256(
            {
                "schema": SCHEMA,
                "q_representation": Q_REPRESENTATION,
                "supercell_hash": self.cell.supercell_hash,
                "cos": self.cos.to_metadata(),
                "sin": self.sin.to_metadata(),
                "cos_amplitude_ang": float(self.cos_amplitude_ang),
                "sin_amplitude_ang": float(self.sin_amplitude_ang),
                "source": self.source,
            }
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.cell.to_metadata(),
            "cos_direction": self.cos.to_metadata(),
            "sin_direction": self.sin.to_metadata(),
            "cos_amplitude_ang": float(self.cos_amplitude_ang),
            "sin_amplitude_ang": float(self.sin_amplitude_ang),
            "complex_mode_hash": self.complex_mode_hash,
            "source": dict(self.source),
        }


def complex_mode(cell: CommensurateCell, mode: PhononMode, *, name: str) -> ComplexMode:
    """Replicate one physical phonon onto the commensurate supercell.

    The per-cell displacement comes from
    :meth:`phonon_provider.PhononMode.displacement_pattern`, which already
    applies the canonical phonon Fourier sign, the atomic-phase convention and
    ``sqrt(hbar / 2 M omega)``. Nothing about phases is re-derived here.
    """
    if not isinstance(mode, PhononMode):
        raise CommensurateError(
            f"expected a phonon_provider.PhononMode, got {type(mode).__name__}: a "
            "synthetic_test_displacement is not a phonon and cannot produce g_{mn nu} (B7)"
        )
    if not mode.conventions.is_canonical:
        raise CommensurateError(
            f"mode conventions {mode.conventions.to_dict()} are not canonical; convert the mode "
            "set with phonon_provider.to_canonical_conventions() at the adapter boundary (S29)"
        )
    if not np.allclose(np.asarray(mode.q_fractional, dtype=np.float64).reshape(3),
                       np.asarray(cell.q_primitive, dtype=np.float64), atol=1e-9):
        raise CommensurateError(
            f"the mode is at q = {np.asarray(mode.q_fractional).reshape(3).tolist()} but the "
            f"supercell was built for q = {list(cell.q_primitive)}"
        )
    if int(np.asarray(mode.eigenvector).shape[0]) != cell.primitive_atom_count:
        raise CommensurateError(
            f"the mode spans {np.asarray(mode.eigenvector).shape[0]} atoms, the primitive cell has "
            f"{cell.primitive_atom_count}"
        )

    pattern = np.concatenate(
        [np.asarray(mode.displacement_pattern(image), dtype=np.complex128) for image in cell.images]
    )
    amplitudes = {"cos": float(np.linalg.norm(pattern.real)), "sin": float(np.linalg.norm(pattern.imag))}
    for half, amplitude in amplitudes.items():
        if amplitude <= AMPLITUDE_TOL_ANG:
            raise CommensurateError(
                f"the {half} half of this mode vanishes (||.||_F = {amplitude:.3e} Ang): the "
                "displacement is real at this q and the Gamma chain covers it directly"
            )
    return ComplexMode(
        cell=cell,
        cos=collective(pattern.real, name=f"{name}__cos"),
        sin=collective(pattern.imag, name=f"{name}__sin"),
        cos_amplitude_ang=amplitudes["cos"],
        sin_amplitude_ang=amplitudes["sin"],
        source={
            "branch": int(mode.branch),
            "frequency_ev": float(mode.frequency_ev),
            "phonon_conventions": mode.conventions.to_dict(),
            "masses_amu": np.asarray(mode.masses_amu, dtype=np.float64).tolist(),
        },
    )


# --------------------------------------------------------------------------- #
# The complex response
# --------------------------------------------------------------------------- #


def _combine(cos_values: Any, sin_values: Any, cos_amplitude: float, sin_amplitude: float):
    return cos_amplitude * np.asarray(cos_values) + 1j * sin_amplitude * np.asarray(sin_values)


@dataclass(frozen=True)
class ComplexModeResponse:
    """``D_H(q)`` and the basis response of one complex mode, from two real ones.

    Both halves must be ordinary Gamma artifacts *of the supercell*: in route
    Q-A the finite ``q`` is geometry, not phase, and a half signed at ``q != 0``
    would mean somebody phased the blocks as well as the pattern.
    """

    mode: ComplexMode
    cos: PaoCovariantResponse
    sin: PaoCovariantResponse

    def __post_init__(self) -> None:
        for half, perturbation, direction in (
            ("cos", self.cos, self.mode.cos),
            ("sin", self.sin, self.mode.sin),
        ):
            if perturbation.direction.to_metadata() != direction.to_metadata():
                raise CommensurateError(
                    f"the {half} perturbation was computed along {perturbation.direction.name!r}, "
                    f"not along the mode's own {direction.name!r} half"
                )
            if any(perturbation.raw.q):
                raise CommensurateError(
                    f"the {half} half is signed at q = {list(perturbation.raw.q)}. In route Q-A "
                    "the artifact q is the supercell's, which is Gamma; the finite q is the "
                    "displacement pattern. A phased block *and* a phased pattern double-counts it"
                )
            if perturbation.raw.direction.atom_count != self.mode.cell.atom_count:
                raise CommensurateError(
                    f"the {half} direction spans {perturbation.raw.direction.atom_count} atoms, "
                    f"the supercell has {self.mode.cell.atom_count}"
                )
        require_same_conventions(
            self.cos.raw.conventions(),
            self.sin.raw.conventions(),
            what="the cos and sin halves of the complex mode",
        )
        if self.cos.formalism_id != self.sin.formalism_id:
            raise CommensurateError(
                f"the halves were built under different formalisms: {self.cos.formalism_id!r} vs "
                f"{self.sin.formalism_id!r}"
            )
        if self.cos.response.representation != self.sin.response.representation:
            raise CommensurateError(
                "the halves carry different basis-response representations: "
                f"{self.cos.response.representation!r} vs {self.sin.response.representation!r}"
            )
        if self.cos.raw.no_u != self.sin.raw.no_u:
            raise CommensurateError(
                f"the halves disagree in size: {self.cos.raw.no_u} vs {self.sin.raw.no_u} orbitals"
            )
        if not np.array_equal(self.cos.raw.isc_off, self.sin.raw.isc_off):
            raise CommensurateError(
                "the halves were serialised over different image tables; the same supercell must "
                "produce the same isc_off for both"
            )

    # -- provenance ----------------------------------------------------------
    @property
    def formalism_id(self) -> str:
        return self.cos.formalism_id

    @property
    def q_primitive(self) -> tuple[float, float, float]:
        return self.mode.cell.q_primitive

    def conventions(self) -> dict[str, Any]:
        """The halves' q-space record, plus what route Q-A adds on top of it."""
        return {
            **self.cos.raw.conventions(),
            "q_representation": Q_REPRESENTATION,
            "q_primitive": list(self.q_primitive),
            "supercell_matrix": self.mode.cell.matrix.tolist(),
            "supercell_hash": self.mode.cell.supercell_hash,
            "complex_mode_hash": self.mode.complex_mode_hash,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "ticket": TICKET,
            "status": CANDIDATE_STATUS,
            "formalism_id": self.formalism_id,
            "convention": self.conventions(),
            "mode": self.mode.to_metadata(),
            "electronic_derivative_backend": self.cos.electronic_derivative_backend,
            "basis_response_backend": self.cos.basis_response_backend,
            "intra_atomic_included": bool(self.cos.intra_atomic_included),
            "halves": {
                "cos": {
                    "raw_derivative": self.cos.raw.artifact_fields(),
                    "basis_response": self.cos.response.artifact_fields(),
                },
                "sin": {
                    "raw_derivative": self.sin.raw.artifact_fields(),
                    "basis_response": self.sin.response.artifact_fields(),
                },
            },
        }

    # -- the physics ---------------------------------------------------------
    def D_H_at_k(self, k: Any) -> np.ndarray:
        """``D_H(q)`` at supercell ``k``: complex, and exact by linearity."""
        return _combine(
            self.cos.raw.at_k(k)["D_H"],
            self.sin.raw.at_k(k)["D_H"],
            self.mode.cos_amplitude_ang,
            self.mode.sin_amplitude_ang,
        )

    def basis_response_at_k(self, k: Any) -> BasisResponse:
        """The same combination of the two basis responses, same amplitudes."""
        cos_response = self.cos.response.at_k(k)
        sin_response = self.sin.response.at_k(k)
        a_cos, a_sin = self.mode.cos_amplitude_ang, self.mode.sin_amplitude_ang
        if cos_response.representation == REPRESENTATION_S_L_S_R:
            return basis_response_S_L_S_R(
                _combine(cos_response.S_left, sin_response.S_left, a_cos, a_sin),
                _combine(cos_response.S_right, sin_response.S_right, a_cos, a_sin),
                intra_atomic_included=bool(
                    cos_response.intra_atomic_included and sin_response.intra_atomic_included
                ),
                backend=cos_response.backend,
            )
        if cos_response.representation == REPRESENTATION_D_S:
            return basis_response_D_S(
                _combine(cos_response.D_S, sin_response.D_S, a_cos, a_sin),
                backend=cos_response.backend,
            )
        raise CommensurateError(
            f"representation {cos_response.representation!r} has no defined linear combination here"
        )

    def pao_covariant_response(self, H_at_k: Any, S_at_k: Any, k: Any) -> np.ndarray:
        """(F1) as an explicit complex matrix. Reference path, it inverts ``S``."""
        return pao_covariant_response(
            self.D_H_at_k(k), H_at_k, S_at_k, self.basis_response_at_k(k)
        )

    def block(self, row: Eigenspace, col: Eigenspace) -> EpcBlock:
        """(F2) between the ``k+q`` window (rows) and the ``k`` window (columns).

        Both windows sit at the *same* supercell ``k`` — that is what folding
        means — so this is the ordinary contraction, on the complex ``D_H(q)``.
        """
        if not np.allclose(row.k, col.k, atol=1e-12):
            raise EpcContractionError(
                f"row k = {row.k} and column k = {col.k} differ. In a commensurate supercell "
                "k and k+q fold onto the same supercell k; two different k mean the windows "
                "did not come from the same solve"
            )
        return epc_matrix_elements(
            self.D_H_at_k(col.k),
            self.basis_response_at_k(col.k),
            C_row=row.C,
            C_col=col.C,
            eps_row=row.eps,
            eps_col=col.eps,
        )


# --------------------------------------------------------------------------- #
# Which states are the k ones and which are the k+q ones
# --------------------------------------------------------------------------- #


def bloch_character_basis(
    orbital_image_index: Any, images: Any, k_primitive: Any, *, orbitals_per_cell: int | None = None
) -> np.ndarray:
    """The symmetry-adapted columns of primitive wavevector ``k`` in the supercell.

    Column ``mu`` is ``e[(mu, l)] = exp(+2i pi k.R_l) / sqrt(N_c)`` (canonical
    electronic Fourier sign, cell gauge). These span the ``k``-character
    subspace exactly, whatever ``S`` is, because translations commute with it —
    which is why the selection below needs no band index and no eigenvalue.
    """
    image_of_orbital = np.asarray(orbital_image_index, dtype=np.int64).reshape(-1)
    image_table = np.asarray(images, dtype=np.float64)
    k = np.asarray(k_primitive, dtype=np.float64).reshape(3)
    n_cells = int(image_table.shape[0])
    no_u = int(image_of_orbital.size)
    per_cell = no_u // n_cells if orbitals_per_cell is None else int(orbitals_per_cell)
    if per_cell * n_cells != no_u:
        raise CommensurateError(
            f"{no_u} orbitals do not split into {n_cells} cells of {per_cell}"
        )
    phases = np.exp(2j * np.pi * (image_table @ k))
    basis = np.zeros((no_u, per_cell), dtype=np.complex128)
    within_cell = np.arange(no_u) % per_cell
    basis[np.arange(no_u), within_cell] = phases[image_of_orbital] / np.sqrt(n_cells)
    return basis


def unfolding_weights(C: Any, S: Any, basis: Any) -> np.ndarray:
    """Weight of each state on one primitive-``k`` character subspace.

    ``w_n = || B~† S C_n ||^2`` with ``B~`` the ``S``-orthonormalised character
    basis. Over the full folding star the weights of a state sum to one, which
    is the check the caller should make instead of trusting any single value.
    """
    columns = np.asarray(basis)
    gram = columns.conj().T @ (np.asarray(S) @ columns)
    eigenvalues, vectors = np.linalg.eigh(gram)
    if eigenvalues.min() <= 0.0:
        raise CommensurateError(
            "the character basis is linearly dependent in the S metric; the orbital/image map "
            "does not describe this supercell"
        )
    orthonormal = columns @ (vectors / np.sqrt(eigenvalues) @ vectors.conj().T)
    overlap = subspace_overlap(orthonormal, np.asarray(S), np.asarray(C))
    return np.real(np.sum(np.abs(overlap) ** 2, axis=0))


def project_character(
    eigenspace: Eigenspace,
    H: Any,
    S: Any,
    basis: Any,
    *,
    label: str,
    occupancy_tol: float = 1e-6,
) -> tuple[Eigenspace, dict[str, Any]]:
    """The part of a window that carries one primitive-``k`` character.

    Selecting *states* would be wrong, and this is the whole reason S30 exists:
    ``k`` and ``k+q`` fold onto the same supercell ``k``, so they are typically
    degenerate there and the solver returns arbitrary mixtures of the two — in
    graphene, every real combination of ``K`` and ``K'`` is an equally valid
    eigenvector. So the window is *projected* onto the character subspace, the
    projection is re-orthonormalised (Loewdin) and ``H`` is re-diagonalised
    inside it (Rayleigh-Ritz), which gives states that are simultaneously
    eigenstates and of definite character.

    The projector's eigenvalues inside the window must be 0 or 1 to
    ``occupancy_tol``: anything in between is a state the window cut in half,
    and its character subspace is not contained in what was solved for.
    """
    C = np.asarray(eigenspace.C)
    S = np.asarray(S)
    H = np.asarray(H)
    columns = np.asarray(basis)
    gram = columns.conj().T @ (S @ columns)
    values, vectors = np.linalg.eigh(gram)
    if values.min() <= 0.0:
        raise CommensurateError("the character basis is linearly dependent in the S metric")
    orthonormal = columns @ (vectors / np.sqrt(values) @ vectors.conj().T)

    overlap = subspace_overlap(orthonormal, S, C)  # M = B~† S C
    occupancy, rotation = np.linalg.eigh(overlap.conj().T @ overlap)
    partial = [
        float(value)
        for value in occupancy
        if occupancy_tol < value < 1.0 - occupancy_tol
    ]
    if partial:
        raise CommensurateError(
            f"the {label!r} character has eigenvalues {partial} inside window "
            f"{eigenspace.label!r}: the window holds only part of a state's character subspace, "
            "so the projection would truncate it. Widen the window"
        )
    keep = [index for index, value in enumerate(occupancy) if value >= 1.0 - occupancy_tol]
    if not keep:
        raise CommensurateError(
            f"window {eigenspace.label!r} carries no {label!r} character at all "
            f"(largest eigenvalue {occupancy.max(initial=0.0):.3e})"
        )
    span = C @ rotation[:, keep]  # in the window, of definite character
    span_gram = span.conj().T @ (S @ span)
    span_values, span_vectors = np.linalg.eigh(span_gram)
    span = span @ (span_vectors / np.sqrt(span_values) @ span_vectors.conj().T)

    # Rayleigh-Ritz inside the character subspace: S-orthonormal by the line above.
    energies, states = np.linalg.eigh(span.conj().T @ (H @ span))
    coefficients = span @ states
    residuals = np.linalg.norm(
        H @ coefficients - (S @ coefficients) * energies[None, :], axis=0
    ) / max(float(np.abs(energies).max(initial=1.0)), 1e-12)
    identity = coefficients.conj().T @ (S @ coefficients) - np.eye(len(keep))
    condition = float(np.linalg.cond(gram))
    resolutions = eigenvalue_resolutions_ev(
        energies, residuals, condition, int(C.shape[0])
    )
    projected = Eigenspace(
        label=f"{eigenspace.label}::{label}",
        k=eigenspace.k,
        C=coefficients,
        eps=energies,
        identity_error=float(np.abs(identity).max(initial=0.0)),
        identity_tolerance=eigenspace.identity_tolerance * PROJECTION_PRODUCTS,
        resolutions_ev=resolutions,
        cluster_labels=near_degenerate_clusters(energies, resolutions),
        node=eigenspace.node,
        diagnostics={**eigenspace.diagnostics, "character": label},
    )
    diagnostics = {
        "character": label,
        "dimension": int(len(keep)),
        "projector_eigenvalues": [float(value) for value in occupancy],
        "occupancy_tolerance": float(occupancy_tol),
        "state_weights_in_window": [
            float(value) for value in unfolding_weights(C, S, columns)
        ],
        "energies_ev": energies.tolist(),
        "maximum_ritz_relative_residual": float(residuals.max(initial=0.0)),
        "identity_maximum_absolute_error": projected.identity_error,
        "identity_tolerance": projected.identity_tolerance,
    }
    return projected, diagnostics


def k_to_k_plus_q_report(
    response: ComplexModeResponse,
    eigenspace: Eigenspace,
    H_at_k: Any,
    S_at_k: Any,
    *,
    k_primitive: Any,
    orbital_image_index: Any,
    occupancy_tol: float = 1e-6,
) -> dict[str, Any]:
    """``g`` from the ``k`` character subspace to the ``k+q`` one, plus its metrics.

    One solve, one eigenspace, two subspaces: ``k`` and ``k+q`` fold together by
    construction, and are separated by Bloch character in the ``S`` metric
    rather than by band index (GO-6). Everything returned is
    ``candidate__pending_go5``.
    """
    cell = response.mode.cell
    k_initial = np.asarray(k_primitive, dtype=np.float64).reshape(3)
    k_final = k_plus_q(k_initial, cell.q_primitive)
    if not cell.folds_together(k_initial, k_final):
        raise CommensurateError(
            f"k = {k_initial.tolist()} and k+q = {k_final.tolist()} do not fold onto the same "
            "supercell k; the supercell is not commensurate with this q after all"
        )
    if not np.allclose(cell.fold(k_initial), np.asarray(eigenspace.k), atol=1e-9):
        raise CommensurateError(
            f"the window is at supercell k = {list(eigenspace.k)} but k = {k_initial.tolist()} "
            f"folds to {cell.fold(k_initial).tolist()}"
        )

    initial, initial_diagnostics = project_character(
        eigenspace,
        H_at_k,
        S_at_k,
        bloch_character_basis(orbital_image_index, cell.images, k_initial),
        label="k",
        occupancy_tol=occupancy_tol,
    )
    final, final_diagnostics = project_character(
        eigenspace,
        H_at_k,
        S_at_k,
        bloch_character_basis(orbital_image_index, cell.images, k_final),
        label="k_plus_q",
        occupancy_tol=occupancy_tol,
    )
    block = response.block(final, initial)
    separation = subspace_metrics(subspace_overlap(final.C, np.asarray(S_at_k), initial.C))
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "status": CANDIDATE_STATUS,
        "k_primitive": k_initial.tolist(),
        "k_plus_q_primitive": k_final.tolist(),
        "k_supercell": list(eigenspace.k),
        "perturbation": response.provenance(),
        "subspace_separation": separation,
        "initial_subspace": initial_diagnostics,
        "final_subspace": final_diagnostics,
        "initial_energies_ev": initial.eps.tolist(),
        "final_energies_ev": final.eps.tolist(),
        "g": {
            "units": block.units,
            "formalism_id": block.formalism_id,
            "representation": block.representation,
            "intra_atomic_included": bool(block.intra_atomic_included),
            "basis_response_backend": block.basis_response_backend,
            "values_real": np.real(block.values).tolist(),
            "values_imaginary": np.imag(block.values).tolist(),
            **gauge_invariant_block_metrics(block),
            "cluster_blocks": block_metrics(
                block.values, final.cluster_labels, initial.cluster_labels
            ),
        },
    }


#: The Gamma full-KS-EPC gate this module's halves inherit verbatim (Sec. 2:
#: "cos and sin stay separate artifacts", ordinary Gamma artifacts of the
#: supercell). Read from docs/epc_go4_graphene_gamma_verdict.md, not measured
#: here: this module runs no physics.
GO4_PHYSICAL_EPC_LEVEL = "NO_GO"
GO4_BLOCKING_GATE = (
    "gate 5 (basis_response_validated_and_required_by_formalism_id): the "
    "intra-atomic term is unresolved (C14C)"
)


def go5_decision() -> dict[str, Any]:
    """E-F_001-S31: what this module's machinery earns, and what it cannot.

    Two layers, judged separately, exactly as GO-4 was:

    * **Algebra layer** -- everything roadmap Fase 7 lists that a toy chain
      can falsify: ``k -> k+q`` by subspace (S30), ``q`` and ``-q``, a
      cell-origin shift, an atom crossing the periodic boundary, and an
      alternative provider atomic-phase convention. All five pass at machine
      precision (this module's tests) -- that is what S31 adds on top of S30.
    * **Physical layer** -- producing ``g`` on real graphene ``K`` needs a
      basis-response artifact for the ``cos``/``sin`` halves, and by
      construction (Sec. 2) those halves are ordinary Gamma artifacts of the
      supercell: the identical backend docs/epc_go4_graphene_gamma_verdict.md
      already put at ``NO_GO`` (its gate 5, the intra-atomic term, C14C).
      Nothing about a finite ``q`` touches that term, so running the same
      backend on a bigger cell would not test anything new about it -- it
      would just re-report the open gate under a different label. No
      graphene-K execution is attempted here for that reason, not for lack of
      SIESTA/Graph2Mat/PhononProvider inputs.

    GO-5 (global) is therefore ``NO_GO``, blocked by GO-4, not by anything
    q-space specific. It flips to attemptable the same day GO-4 does.
    """
    return {
        "schema": SCHEMA,
        "ticket": "E-F_001-S31",
        "memo": "docs/epc_s31_graphene_k_go5_verdict.md",
        "algebra_layer": "PASS",
        "physical_layer": "NOT_ATTEMPTED",
        "go5": "NO_GO",
        "blocked_by": "GO-4 full-KS-EPC-level (docs/epc_go4_graphene_gamma_verdict.md)",
        "blocking_gate": GO4_BLOCKING_GATE,
        "checks": {
            "k_to_k_plus_q_by_subspace": "PASS",
            "q_and_minus_q": "PASS",
            "cell_origin_shift": "PASS",
            "atom_across_periodic_boundary": "PASS",
            "alternative_atomic_phase_convention": "PASS",
        },
    }


def commensurate_contract() -> dict[str, Any]:
    """The S30 contract as data, for manifests and the UI (no physics here)."""
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": "docs/epc_s30_graphene_k_conmensurable.md",
        "q_representation": Q_REPRESENTATION,
        "convention_id": CONVENTION_ID,
        "status": CANDIDATE_STATUS,
        "rules": {
            "artifact_q": "the two halves are Gamma artifacts of the supercell; the finite q is "
            "the displacement pattern, never a phase applied to Gamma blocks",
            "halves": "cos and sin are separately computed and separately signed; the complex "
            "mode is their linear combination with the two Frobenius amplitudes in Ang",
            "k_plus_q": "k and k+q fold onto the same supercell k and are separated by primitive "
            "Bloch character in the S metric, not by band index",
            "promotion": "everything produced here stays candidate__pending_go5",
        },
    }
