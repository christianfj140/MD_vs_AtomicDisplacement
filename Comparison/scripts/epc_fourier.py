#!/usr/bin/env python3
"""D01 / E-F_001-S29: the one q-space convention, and the adapters into it.

Every phase in the EPC chain is fixed here and nowhere else: ``R``, ``tau``,
block orientation, the electronic Fourier sign, the phonon Fourier sign, the
atomic phase, ``k+q`` and ``q <-> -q``. Modules that used to spell a convention
out in a docstring string constant now import it (``epc_basis_response``,
``compute_epc_matrix_elements``, ``phonon_provider``), so there is exactly one
place where a sign can be wrong.

The canon (:data:`CANONICAL`, ``epc_qspace_canonical_v1``)
---------------------------------------------------------

* **R** — integer lattice images ``l``; ``R_l = l @ cell``, rows of ``cell`` are
  ``a1,a2,a3``. ``k`` and ``q`` are fractional in the dual basis, so
  ``k . R_l = 2 pi (k_frac . l)``.
* **tau** — atomic positions, and *which* image is the home one is part of the
  geometry artifact, not a free choice: see :func:`relabel_home_images`.
* **block orientation** ``row_home_cell`` —
  ``X_{mu,nu}(R_l) = <phi_mu(0)| x |phi_nu(R_l)>``: the row orbital sits in the
  home cell, the column orbital in the image.
* **electronic Fourier sign** ``+1`` — ``X(k) = sum_l exp(+2i pi k.R_l) X(R_l)``.
* **phonon Fourier sign** ``+1`` — ``u_{l kappa} = e_kappa(q) exp(+2i pi q.R_l)``.
* **atomic phase** ``cell`` — no ``exp(i k.tau_mu)`` on the orbitals and none on
  the phonon eigenvector. Both live in the *cell* gauge, which is why the same
  ``exp(+2i pi q.R)`` appears on the electronic and the phonon side and no
  relative factor survives between them.
* **k+q** — the final state is at ``k + q`` in the same fractional basis,
  unfolded. Folding by a reciprocal lattice vector ``G`` is the identity in the
  cell gauge (``exp(2i pi G.R) = 1`` for integer ``R``); it is *not* in the atom
  gauge, where it is ``diag(exp(2i pi G.tau))``.
* **q <-> -q** — no separate convention: with real real-space blocks
  ``X(-q) = conj(X(q))`` and ``e(-q) = conj(e(q))``. :func:`minus_q_matrix` is
  that identity, guarded by its precondition instead of asserted.

Two gauges, and what each one is invariant under
------------------------------------------------

The cell gauge is *not* invariant when an atom is re-expressed in another
periodic image (``tau -> tau + T``): the blocks re-index and ``X(k)`` picks up
``W X W^dag`` with ``W = diag(exp(2i pi k.T_mu))``. The atom gauge is invariant
under that, and picks up an (irrelevant) global phase under an origin shift
instead. Both are congruences, so generalized eigenvalues never move — which is
exactly the test :func:`relabel_home_images` and :func:`image_relabel_unitary`
exist to make runnable.

Adapters
--------

An external provider declares a :class:`QSpaceConventions` and is converted by
:func:`to_canonical_at_k` / :func:`to_canonical_vector_at_k`, whose inverses
(:func:`from_canonical_at_k`, ``inverse=True``) round-trip at machine
precision. There is no per-backend phase patch anywhere: an unknown backend
raises in :func:`external_conventions` rather than getting a hand-written
``exp(...)`` somewhere downstream.

One subtlety the field names hide. For *real-space blocks* only the block
orientation matters — the blocks are the matrix elements themselves, and a
Fourier sign or an atomic phase is a statement about ``X(k)``, not about
``X(R)``. So :func:`matrix_at_k` (blocks in) uses the orientation alone, while
:func:`to_canonical_at_k` (an already-summed ``X(k)`` in) uses all three. In
``X(k)`` orientation and sign are the same operation: with
``X'_{mu,nu}(R) = <phi_mu(R)|x|phi_nu(0)> = X_{mu,nu}(-R)`` a column-home table
summed with ``+1`` is the row-home table summed with ``-1``, which is why only
the product :func:`effective_fourier_sign` ever enters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CONVENTION_ID = "epc_qspace_canonical_v1"

GAMMA = (0.0, 0.0, 0.0)

FOURIER_SIGN_CANONICAL = 1

ATOMIC_PHASE_CELL = "cell"
ATOMIC_PHASE_ATOM = "atom"
ATOMIC_PHASES = (ATOMIC_PHASE_CELL, ATOMIC_PHASE_ATOM)

BLOCK_ROW_HOME = "row_home_cell"
BLOCK_COLUMN_HOME = "column_home_cell"
BLOCK_ORIENTATIONS = (BLOCK_ROW_HOME, BLOCK_COLUMN_HOME)

# The prose forms, kept because manifests and the UI already publish them.
BLOCK_CONVENTION = "X_{mu,nu}(R_l) = <phi_mu(0)|x|phi_nu(R_l)>, R_l integer lattice vector"
BLOCH_CONVENTION = (
    "X(k) = sum_l exp(+2i*pi*k.R_l) X(R_l); k fractional; cell gauge, no exp(i k.tau) "
    "on the atomic positions (memo section 2)"
)
ATOMIC_PHASE_CONVENTION = (
    "u_{l,kappa} = e_kappa(q) exp(+i q.R_l); the exp(i q.tau_kappa) factor stays outside the "
    "eigenvector and is declared by the PhononProvider (memo section 8)"
)
K_PLUS_Q_CONVENTION = (
    "final state at k+q, unfolded fractional coordinates; folding by a reciprocal lattice "
    "vector G is the identity in the cell gauge and diag(exp(2i*pi*G.tau)) in the atom gauge"
)
MINUS_Q_CONVENTION = (
    "X(-q) = conj(X(q)) and e(-q) = conj(e(q)) whenever the real-space blocks are real; "
    "no independent sign convention for -q"
)


class QConventionError(ValueError):
    """A q-space object was used outside the convention it declares."""


@dataclass(frozen=True)
class QSpaceConventions:
    """What a provider means by its phases. Declared, never inferred."""

    fourier_sign: int = FOURIER_SIGN_CANONICAL
    phonon_fourier_sign: int = FOURIER_SIGN_CANONICAL
    atomic_phase: str = ATOMIC_PHASE_CELL
    block_orientation: str = BLOCK_ROW_HOME

    def __post_init__(self) -> None:
        for name in ("fourier_sign", "phonon_fourier_sign"):
            if getattr(self, name) not in (1, -1):
                raise QConventionError(f"{name} must be +1 or -1, got {getattr(self, name)!r}")
        if self.atomic_phase not in ATOMIC_PHASES:
            raise QConventionError(f"atomic_phase must be one of {ATOMIC_PHASES}")
        if self.block_orientation not in BLOCK_ORIENTATIONS:
            raise QConventionError(f"block_orientation must be one of {BLOCK_ORIENTATIONS}")

    @property
    def is_canonical(self) -> bool:
        return self == CANONICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "convention_id": CONVENTION_ID if self.is_canonical else "external",
            "fourier_sign": int(self.fourier_sign),
            "phonon_fourier_sign": int(self.phonon_fourier_sign),
            "atomic_phase": self.atomic_phase,
            "block_orientation": self.block_orientation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QSpaceConventions":
        return cls(
            fourier_sign=int(payload["fourier_sign"]),
            phonon_fourier_sign=int(payload.get("phonon_fourier_sign", payload["fourier_sign"])),
            atomic_phase=str(payload.get("atomic_phase", ATOMIC_PHASE_CELL)),
            block_orientation=str(payload.get("block_orientation", BLOCK_ROW_HOME)),
        )


CANONICAL = QSpaceConventions()

# Declared conventions of the providers this repository actually reads. An
# adapter is an entry here plus the generic conversion below; a backend that is
# not in this table is a missing declaration, not a licence to patch a phase.
EXTERNAL_CONVENTIONS: dict[str, QSpaceConventions] = {
    # sisl/SIESTA hand back H(R), S(R) in the cell gauge with exp(+i k.R).
    "sisl": CANONICAL,
    "siesta": CANONICAL,
    "graph2mat": CANONICAL,
    # VIBRA is diagonalised in the same cell gauge; its cm^-1 and its 3N ordering
    # are units/layout, handled by phonon_provider, not phases.
    "siesta_vibra": CANONICAL,
    # Phonopy's default dynamical matrix carries exp(i q.(R + tau' - tau)).
    "phonopy_atom_gauge": QSpaceConventions(atomic_phase=ATOMIC_PHASE_ATOM),
    # DFTBephy writes the atomic phase into the displacement explicitly
    # (roadmap Fase 7.4 / Croy et al. 2023).
    "dftbephy": QSpaceConventions(atomic_phase=ATOMIC_PHASE_ATOM),
}


def external_conventions(backend: str) -> QSpaceConventions:
    """The declared conventions of ``backend``; unknown backends raise."""
    try:
        return EXTERNAL_CONVENTIONS[backend]
    except KeyError:
        raise QConventionError(
            f"no declared q-space conventions for backend {backend!r}. Add its conventions to "
            f"EXTERNAL_CONVENTIONS (known: {sorted(EXTERNAL_CONVENTIONS)}); a per-backend phase "
            f"correction applied at the call site is what this module exists to prevent"
        ) from None


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def _vector(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise QConventionError(f"{name} must be a fractional 3-vector, got {value!r}")
    return array


def effective_fourier_sign(conventions: QSpaceConventions = CANONICAL) -> int:
    """The only combination of sign and orientation that reaches ``X(k)``."""
    return int(conventions.fourier_sign) * (
        1 if conventions.block_orientation == BLOCK_ROW_HOME else -1
    )


def atomic_phase(
    k: Any, positions_frac: Any, sign: int = FOURIER_SIGN_CANONICAL
) -> np.ndarray:
    """``exp(i * sign * k . tau)`` per position — the cell/atom gauge unitary."""
    tau = np.asarray(positions_frac, dtype=np.float64).reshape(-1, 3)
    return np.exp(2j * np.pi * int(sign) * (tau @ _vector("k", k)))


def fractional_positions(positions_ang: Any, cell_ang: Any) -> np.ndarray:
    """Cartesian positions to fractional coordinates (rows of ``cell`` = ``a_i``)."""
    cell = np.asarray(cell_ang, dtype=np.float64)
    if cell.shape != (3, 3):
        raise QConventionError(f"cell must be 3x3, got shape {cell.shape}")
    return np.asarray(positions_ang, dtype=np.float64).reshape(-1, 3) @ np.linalg.inv(cell)


def bloch_sum(
    blocks: Any, isc_off: Any, k: Any = GAMMA, sign: int = FOURIER_SIGN_CANONICAL
) -> np.ndarray:
    """``X(k) = sum_l exp(2i pi sign k.R_l) X(R_l)`` in the cell gauge."""
    phases = np.exp(
        2j * np.pi * int(sign) * (np.asarray(isc_off, dtype=np.float64) @ _vector("k", k))
    )
    return np.tensordot(phases, np.asarray(blocks), axes=(0, 0))


def matrix_at_k(
    blocks: Any, isc_off: Any, k: Any = GAMMA, conventions: QSpaceConventions = CANONICAL
) -> np.ndarray:
    """Canonical ``X(k)`` from real-space blocks written in ``conventions``.

    Only the block orientation can matter here: a Fourier sign and an atomic
    phase are statements about ``X(k)``, not about the real-space matrix
    elements, so they are deliberately ignored (module docstring).
    """
    orientation = 1 if conventions.block_orientation == BLOCK_ROW_HOME else -1
    return bloch_sum(blocks, isc_off, k, sign=orientation)


def gauge_unitary(k: Any, positions_frac: Any) -> np.ndarray:
    """``U = diag(exp(+2i pi k.tau))``: ``X_cell = U X_atom U^dag`` in the canon.

    The canonical sign is the right one here even for a provider whose sign is
    ``-1``, because the sign is undone first (:func:`to_canonical_at_k`).
    """
    return atomic_phase(k, positions_frac, FOURIER_SIGN_CANONICAL)


def _gauge_steps(conventions: QSpaceConventions, *, phonon: bool = False) -> tuple[bool, bool]:
    sign = int(conventions.phonon_fourier_sign) if phonon else effective_fourier_sign(conventions)
    return sign != FOURIER_SIGN_CANONICAL, conventions.atomic_phase == ATOMIC_PHASE_ATOM


def to_canonical_at_k(
    matrix: Any,
    k: Any,
    conventions: QSpaceConventions = CANONICAL,
    *,
    orbital_positions_frac: Any = None,
    real_space_blocks_real: bool = True,
    inverse: bool = False,
) -> np.ndarray:
    """Convert an already-summed ``X(k)`` into the canon (or back, ``inverse``).

    Two explicit steps, in this order: conjugate if the provider's effective
    sign is ``-1``, then undo the atom gauge with ``U X U^dag``. The
    conjugation is only the right way to flip the sign of a summed ``X(k)`` when
    the real-space blocks are real — say ``real_space_blocks_real=False`` and
    convert the blocks with :func:`matrix_at_k` instead (spin-orbit, external
    magnetic field).
    """
    x = np.asarray(matrix)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise QConventionError(f"expected a square X(k), got shape {x.shape}")
    flip, atom = _gauge_steps(conventions)
    if flip and not real_space_blocks_real:
        raise QConventionError(
            "flipping the Fourier sign of an already-summed X(k) is a conjugation, which is "
            "only the right operation when the real-space blocks are real. Convert the blocks "
            "with matrix_at_k() instead"
        )
    unitary = None
    if atom:
        if orbital_positions_frac is None:
            raise QConventionError(
                "the atom gauge needs orbital_positions_frac (one fractional tau per orbital)"
            )
        unitary = gauge_unitary(k, orbital_positions_frac)
        if unitary.size != x.shape[0]:
            raise QConventionError(
                f"orbital_positions_frac has {unitary.size} rows, X(k) has {x.shape[0]}"
            )
    if inverse:
        if atom:
            x = unitary.conj()[:, None] * x * unitary[None, :]
        return np.conj(x) if flip else x
    if flip:
        x = np.conj(x)
    if atom:
        x = unitary[:, None] * x * unitary.conj()[None, :]
    return x


def from_canonical_at_k(
    matrix: Any,
    k: Any,
    conventions: QSpaceConventions = CANONICAL,
    *,
    orbital_positions_frac: Any = None,
    real_space_blocks_real: bool = True,
) -> np.ndarray:
    """The exact inverse of :func:`to_canonical_at_k` (round-trip is exact)."""
    return to_canonical_at_k(
        matrix,
        k,
        conventions,
        orbital_positions_frac=orbital_positions_frac,
        real_space_blocks_real=real_space_blocks_real,
        inverse=True,
    )


def to_canonical_vector_at_k(
    vector: Any,
    k: Any,
    conventions: QSpaceConventions = CANONICAL,
    *,
    positions_frac: Any = None,
    phonon: bool = False,
    inverse: bool = False,
) -> np.ndarray:
    """Convert eigenvector columns (electronic) or ``e_kappa(q)`` (phonon).

    Same two steps as :func:`to_canonical_at_k`, in the same order, because
    ``X_cell = U X_atom U^dag`` gives ``C_cell = U C_atom``. ``phonon=True``
    reads ``phonon_fourier_sign`` and leaves the block orientation out of it,
    since a phonon eigenvector has no row/column to orient.
    """
    array = np.asarray(vector)
    flip, atom = _gauge_steps(conventions, phonon=phonon)
    unitary = None
    if atom:
        if positions_frac is None:
            raise QConventionError("the atom gauge needs positions_frac")
        unitary = gauge_unitary(k, positions_frac)
        if unitary.size != array.shape[0]:
            raise QConventionError(
                f"positions_frac has {unitary.size} rows, the vector has {array.shape[0]}"
            )
        unitary = unitary.reshape((-1,) + (1,) * (array.ndim - 1))
    if inverse:
        if atom:
            array = unitary.conj() * array
        return np.conj(array) if flip else array
    if flip:
        array = np.conj(array)
    if atom:
        array = unitary * array
    return array


# --------------------------------------------------------------------------- #
# k, q, and the periodic image an atom is written in
# --------------------------------------------------------------------------- #


def k_plus_q(k: Any, q: Any) -> np.ndarray:
    """The final-state wavevector, unfolded (:data:`K_PLUS_Q_CONVENTION`)."""
    return _vector("k", k) + _vector("q", q)


def fold_to_first_bz(k: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(k_folded, G)`` with ``k = k_folded + G``, ``k_folded`` in ``[-1/2, 1/2)``."""
    vector = _vector("k", k)
    g = np.floor(vector + 0.5)
    return vector - g, g.astype(np.int64)


def minus_q_matrix(matrix: Any, *, real_space_blocks_real: bool = True) -> np.ndarray:
    """``X(-q)`` from ``X(q)`` (:data:`MINUS_Q_CONVENTION`)."""
    if not real_space_blocks_real:
        raise QConventionError(
            "X(-q) = conj(X(q)) assumes real real-space blocks; Bloch sum at -q instead"
        )
    return np.conj(np.asarray(matrix))


def image_relabel_unitary(k: Any, orbital_image_shifts: Any) -> np.ndarray:
    """``W_mu = exp(2i pi k.T_mu)`` for orbitals rewritten in image ``T_mu``.

    ``X'(k) = W X(k) W^dag`` in the cell gauge; the atom gauge is invariant,
    because ``tau_mu -> tau_mu + T_mu`` absorbs exactly this factor.
    """
    shifts = np.asarray(orbital_image_shifts, dtype=np.float64).reshape(-1, 3)
    return np.exp(2j * np.pi * (shifts @ _vector("k", k)))


def relabel_home_images(blocks: Any, isc_off: Any, orbital_image_shifts: Any) -> np.ndarray:
    """Rewrite each orbital in another periodic image: an atom crossing a face.

    ``X'_{mu,nu}(R_l) = X_{mu,nu}(R_l + T_nu - T_mu)``, on the same image table.

    Blocks outside the table are zero — that is what the auxiliary supercell
    means — so the only way this loses information is when a *stored* non-zero
    block would have to land outside it. That raises, instead of vanishing.
    """
    values = np.asarray(blocks)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise QConventionError(f"blocks must be (n_s, no_u, no_u), got shape {values.shape}")
    isc = np.asarray(isc_off, dtype=np.int64).reshape(-1, 3)
    if isc.shape[0] != values.shape[0]:
        raise QConventionError(f"blocks carry {values.shape[0]} images, isc_off has {isc.shape[0]}")
    shifts = np.asarray(orbital_image_shifts, dtype=np.int64).reshape(-1, 3)
    if shifts.shape[0] != values.shape[1]:
        raise QConventionError(
            f"orbital_image_shifts has {shifts.shape[0]} rows, the blocks have {values.shape[1]} "
            "orbitals (one integer image per orbital, i.e. per its atom)"
        )

    index = {tuple(image): position for position, image in enumerate(isc.tolist())}
    unique = sorted({tuple(row) for row in shifts.tolist()})
    groups = {
        shift: np.flatnonzero((shifts == np.array(shift)).all(axis=1)) for shift in unique
    }
    out = np.zeros_like(values)
    for row_shift, rows in groups.items():
        for column_shift, columns in groups.items():
            delta = np.array(column_shift) - np.array(row_shift)
            grid = np.ix_(rows, columns)
            for source, image in enumerate(isc.tolist()):
                # X'(R) = X(R + delta), so the block stored at R + delta moves to R.
                destination = index.get(tuple(np.array(image) - delta))
                if destination is None:
                    dropped = float(np.abs(values[(source,) + grid]).max(initial=0.0))
                    if dropped > 0.0:
                        raise QConventionError(
                            f"the block at image {tuple(image)} would move to "
                            f"{tuple(np.array(image) - delta)}, which is outside the table, and it "
                            f"is not zero ({dropped:.3e}): the auxiliary supercell is not closed "
                            f"under this relabeling"
                        )
                    continue
                out[(destination,) + grid] = values[(source,) + grid]
    return out


# --------------------------------------------------------------------------- #
# What artifacts declare
# --------------------------------------------------------------------------- #


def convention_fields(
    q: Any = GAMMA,
    *,
    q_aware_source: bool = False,
    conventions: QSpaceConventions = CANONICAL,
) -> dict[str, Any]:
    """The q-space block every EPC artifact carries in its signature.

    The same function feeds ``raw_derivative``, ``basis_response`` and
    ``pao_projected_mode_coupling``, so "same conventions" is an equality of dicts
    rather than a claim in a docstring.
    """
    if not conventions.is_canonical:
        raise QConventionError(
            f"artifacts are signed in {CONVENTION_ID}; convert {conventions.to_dict()} at the "
            "adapter boundary with to_canonical_at_k()/to_canonical_vector_at_k() first"
        )
    values = tuple(float(value) for value in _vector("q", q))
    if any(values) and not q_aware_source:
        raise QConventionError(
            f"q = {values} but the source is not q-aware: a cell-periodic (q = 0) response "
            "cannot be relabelled at finite q (GO-8b)"
        )
    return {
        "convention_id": CONVENTION_ID,
        "blocks": BLOCK_CONVENTION,
        "bloch": BLOCH_CONVENTION,
        "atomic_phase": ATOMIC_PHASE_CONVENTION,
        "k_plus_q": K_PLUS_Q_CONVENTION,
        "minus_q": MINUS_Q_CONVENTION,
        "fourier_sign": int(conventions.fourier_sign),
        "phonon_fourier_sign": int(conventions.phonon_fourier_sign),
        "block_orientation": conventions.block_orientation,
        "q": list(values),
        "q_aware_source": bool(q_aware_source),
    }


_CONVENTION_KEYS = (
    "convention_id",
    "fourier_sign",
    "phonon_fourier_sign",
    "block_orientation",
    "atomic_phase",
    "q",
    "q_aware_source",
)


def require_same_conventions(left: Any, right: Any, *, what: str = "two artifacts") -> None:
    """Refuse a contraction whose inputs were phased differently."""
    mismatches = {
        key: (dict(left).get(key), dict(right).get(key))
        for key in _CONVENTION_KEYS
        if dict(left).get(key) != dict(right).get(key)
    }
    if mismatches:
        raise QConventionError(
            f"{what} do not share the q-space convention: {mismatches}. Convert at the adapter "
            f"boundary; there is no per-backend phase correction downstream"
        )


def qspace_contract() -> dict[str, Any]:
    """The S29 contract as data, for manifests and the UI (no physics here)."""
    return {
        "schema": "epc_qspace_convention_v1",
        "ticket": "D01 / E-F_001-S29",
        "memo": "docs/epc_s29_convenciones_qspace.md",
        "convention_id": CONVENTION_ID,
        "canonical": CANONICAL.to_dict(),
        "definitions": {
            "R": "integer lattice image l; R_l = l @ cell, rows of cell are a1,a2,a3",
            "tau": "atomic position; the home image is fixed by the geometry artifact",
            "blocks": BLOCK_CONVENTION,
            "bloch": BLOCH_CONVENTION,
            "atomic_phase": ATOMIC_PHASE_CONVENTION,
            "k_plus_q": K_PLUS_Q_CONVENTION,
            "minus_q": MINUS_Q_CONVENTION,
        },
        "gauge_covariance": {
            "origin_shift": "cell gauge invariant; atom gauge picks up a global phase that "
            "cancels in U X U^dag",
            "image_relabeling": "cell gauge X'(k) = W X(k) W^dag with W = diag(exp(2i pi k.T)); "
            "atom gauge invariant; generalized eigenvalues invariant in both",
            "bz_folding": "identity in the cell gauge, diag(exp(2i pi G.tau)) in the atom gauge",
        },
        "adapters": {name: value.to_dict() for name, value in EXTERNAL_CONVENTIONS.items()},
        "adapter_policy": "external -> explicit unitary/conjugation -> canonical, with an exact "
        "round trip; an undeclared backend raises in external_conventions()",
    }
