#!/usr/bin/env python3
"""C14B / E-F_001-S16 and C14C / E-F_001-S17: the basis-response provider and its gate.

``epc_formalism.py`` says *what* the moving-PAO formalism needs; this module is
where that object comes from. For one displacement pattern (a Gamma direction
or, later, a ``q`` mode) it delivers exactly the representation
``formalism_id`` demands, in canonical units, with the provenance that lets a
downstream ``pao_covariant_response`` be signed:

    moving_pao_projected_dual_basis_v1  ->  S_L and S_R, separately
    moving_pao_symmetric_overlap_v1     ->  D_S is enough (degraded, memo 5)
    moving_pao_ignore_overlap_v1        ->  takes no basis response at all

The representation is chosen by the derivation, not by what a backend happens
to have lying around: :class:`BasisResponseArtifact` refuses to serve a ``D_S``
response to the production formalism, and there is no code path anywhere here
that splits a ``D_S`` back into ``S_L``/``S_R`` (memo 4.7 shows two systems with
identical ``D_S`` and different ``g``).

Sources, in the order the memo (section 6) recommends:

* :func:`from_S_L_S_R` — an analytic PAO overlap derivative. Native ``S_L``/``S_R``,
  no SCF, scalable. Not implemented yet; this is the door it comes through.
* :func:`from_dhsdr` — SIESTA ``dS/dR`` resolved per displaced atom. The
  inter-atomic split is exact (moving atom ``I`` selects ``S_L`` on its rows and
  ``S_R`` on its columns); the intra-atomic term is *identically invisible* in
  ``dS/dR`` and is therefore missing, which the artifact says out loud in
  ``method`` and in ``intra_atomic_included`` rather than by silence (memo A2).
* :func:`from_covariant_connection` — ``Gamma = S^-1 S_R``. Equivalent to the
  pair given ``S`` (``S_R = S Gamma``, ``S_L = Gamma^dag S``, memo 4.2), so it
  satisfies the production formalism, and is signed as its own representation.
* :func:`from_D_S` — frozen finite differences of ``S``. Degraded on purpose.

Everything is stored as real-space blocks ``(n_s, no_u, no_u)`` indexed by the
same image table as ``H`` and ``S``; :meth:`BasisResponseArtifact.at_k` Bloch
sums them in the repository's cell gauge and hands back the
:class:`epc_formalism.BasisResponse` the contraction consumes. Units are
``1/Ang`` throughout.

Backend honesty: the artifact carries both its own backend and the backend of
the ``D_H`` it is meant to be paired with, so ``epc_backend_class`` comes out
``hybrid`` by arithmetic — a Graph2Mat ``D_H`` with a SIESTA overlap response is
never labelled "Graph2Mat EPC".

C14C — the validation gate
--------------------------

:func:`validate_basis_response` is the independent gate C19 must clear before a
``PAO-covariant response`` is built from any of the above. It answers three questions, in
this order, with tolerances pre-registered in this module and never tuned to the
data:

1. *does the response reproduce the best reference available?* — the sum
   ``S_L + S_R`` against an independently measured ``D_S`` (a central finite
   difference of ``S``, or an analytic one), inside the GO-2 noise floor
   ``tau_FD``; the exact identity ``S_L(k) = S_R(k)^dag``; and, for a rigid
   translation, a vanishing sum with a non-vanishing ``A``;
2. *how large is what each approximation drops?* — the response's own
   contribution to ``g``, split into diagonal and interband, and the deltas of
   ``moving_pao_symmetric_overlap_v1`` (drops ``A``) and
   ``moving_pao_ignore_overlap_v1`` (drops everything), both labelled
   ``approximation`` and never truth;
3. *is anything still unresolved?* — the intra-atomic term of ``S_L``/``S_R``
   (memo assumption A2) is invisible to ``dS/dR`` and has no analytic
   implementation yet, so a response that omits it returns ``NO_GO``.
   :func:`intra_atomic_sensitivity` reports how far ``g`` moves per unit of that
   term; that is a *sensitivity*, and this module never turns it into a
   correction factor.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
_SHARED_DIR = SCRIPT_DIR.parents[1] / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from artifact_signature import (  # noqa: E402
    BASIS_RESPONSE_REPRESENTATIONS,
    EPC_BACKEND_CLASS_HYBRID,
    epc_artifact_node,
)
from epc_subspaces import random_gauge_rotation  # noqa: E402

# The q-space conventions live in one module (D01 / E-F_001-S29); these are
# re-exported because manifests and downstream imports already read them here.
from epc_fourier import (  # noqa: E402,F401
    ATOMIC_PHASE_CONVENTION,
    BLOCH_CONVENTION,
    BLOCK_CONVENTION,
    CONVENTION_ID,
    GAMMA,
    bloch_sum,
    convention_fields,
)
from epc_formalism import (  # noqa: E402
    DEGRADED_FORMALISM_IDS,
    FORMALISM_ID,
    FORMALISM_ID_IGNORE_OVERLAP,
    FORMALISM_ID_SYMMETRIC,
    REPRESENTATION_D_S,
    REPRESENTATION_S_L_S_R,
    BasisResponse,
    basis_response_D_S,
    basis_response_S_L_S_R,
    epc_matrix_elements,
)

REPRESENTATION_COVARIANT = "covariant_basis_connection"

# Which representations actually satisfy which formalism. This table is the
# derivation (memo sections 5 and 6), not a preference: D_S fixes only the
# hermitian part of S*Gamma, so it cannot feed (F2).
REQUIRED_REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    FORMALISM_ID: (REPRESENTATION_S_L_S_R, REPRESENTATION_COVARIANT),
    FORMALISM_ID_SYMMETRIC: (REPRESENTATION_D_S, REPRESENTATION_S_L_S_R, REPRESENTATION_COVARIANT),
    FORMALISM_ID_IGNORE_OVERLAP: (),
}

UNITS = "1/Ang"

# What the SIESTA dS/dR split can and cannot see (memo section 6).
METHOD_DHSDR_INTER_ATOMIC = "dhsdr_atom_resolved_split__inter_atomic_only__intra_atomic_omitted"

# |dS/dR| on same-atom blocks should be numerical zero; anything above this and
# the atom -> orbital map or the sign convention is wrong, not the physics.
INTRA_ATOMIC_RESIDUAL_TOL = 1e-10


class BasisResponseError(ValueError):
    """A basis-response artifact was requested that the formalism cannot use."""


# ---------------------------------------------------------------------------
# Where the response came from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseContext:
    """The provenance every basis-response artifact must carry.

    ``electronic_derivative_backend`` is not decoration: it is what makes
    ``epc_backend_class`` computable at the point the response is built, so a
    hybrid can never be mislabelled downstream.
    """

    geometry_signature: str
    basis_signature: str  # orbital_contract_hash (C03)
    electronic_derivative_backend: str
    direction: Any = None  # fd_perturbation_space.Direction
    perturbation_definition: Any = None  # free-form, when there is no Direction
    q: tuple[float, float, float] = GAMMA
    q_aware_source: bool = False
    source: dict[str, Any] = field(default_factory=dict)
    formalism_id: str = FORMALISM_ID

    def __post_init__(self) -> None:
        for name in ("geometry_signature", "basis_signature", "electronic_derivative_backend"):
            if not str(getattr(self, name) or "").strip():
                raise BasisResponseError(f"{name} must be declared; an unsigned response is unusable")
        if (self.direction is None) == (self.perturbation_definition is None):
            raise BasisResponseError(
                "give exactly one of direction=<fd_perturbation_space.Direction> or "
                "perturbation_definition=<what the pattern is>"
            )
        q = tuple(float(value) for value in self.q)
        if len(q) != 3:
            raise BasisResponseError(f"q must be a 3-vector in fractional units, got {self.q!r}")
        object.__setattr__(self, "q", q)
        if any(q) and not self.q_aware_source:
            # memo section 8 / GO-8b: a Gamma-periodic response relabelled q != 0
            # is exactly the silent substitution the roadmap forbids.
            raise BasisResponseError(
                f"q = {q} but the source is not q-aware. A response built from a cell-periodic "
                "(q = 0) perturbation cannot be declared at finite q; build S_L/S_R with the same "
                "exp(+i q.R) phase as D_H, or leave q at Gamma"
            )

    @property
    def definition(self) -> Any:
        """What went into the signature as ``perturbation_definition``."""
        if self.direction is not None:
            return self.direction.to_metadata()
        return self.perturbation_definition


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


def _blocks(name: str, array: Any, *, no_u: int | None = None) -> np.ndarray:
    """Real-space blocks as ``(n_s, no_u, no_u)``; a bare matrix is ``n_s = 1``."""
    values = np.asarray(array)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise BasisResponseError(
            f"{name} must be (n_s, no_u, no_u) blocks or one square matrix, got shape {values.shape}"
        )
    if no_u is not None and values.shape[1] != no_u:
        raise BasisResponseError(f"{name} has no_u = {values.shape[1]}, expected {no_u}")
    if not np.all(np.isfinite(values)):
        raise BasisResponseError(f"{name} has non-finite entries")
    target = np.complex128 if np.issubdtype(values.dtype, np.complexfloating) else np.float64
    return values.astype(target, copy=False)


@dataclass(frozen=True)
class BasisResponseArtifact:
    """One displacement pattern's overlap/basis response, signed and unit-bearing.

    ``blocks`` holds the representation's matrices in real space:
    ``S_left``/``S_right`` for ``S_L_S_R``, ``D_S`` for ``D_S``, ``Gamma`` and
    ``S`` for ``covariant_basis_connection``. Use :meth:`at_k` to get the object
    ``epc_formalism`` consumes.
    """

    representation: str
    blocks: dict[str, np.ndarray]
    isc_off: np.ndarray
    backend: str
    method: str
    intra_atomic_included: bool
    context: ResponseContext
    diagnostics: dict[str, Any] = field(default_factory=dict)
    units: str = UNITS

    def __post_init__(self) -> None:
        if self.representation not in BASIS_RESPONSE_REPRESENTATIONS:
            raise BasisResponseError(
                f"representation must be one of {BASIS_RESPONSE_REPRESENTATIONS}, "
                f"got {self.representation!r}"
            )
        formalism_id = self.context.formalism_id
        allowed = REQUIRED_REPRESENTATIONS.get(formalism_id)
        if allowed is None:
            raise BasisResponseError(
                f"unknown formalism_id {formalism_id!r}; known: {sorted(REQUIRED_REPRESENTATIONS)}"
            )
        if not allowed:
            raise BasisResponseError(
                f"{formalism_id!r} takes no basis response at all — it is the control negative "
                "g = C^dag D_H C. Build it with epc_matrix_elements(..., control_negative=True)"
            )
        if self.representation not in allowed:
            raise BasisResponseError(
                f"{formalism_id!r} needs one of {allowed}, and {self.representation!r} does not "
                "determine it. D_S fixes only the hermitian part of S*Gamma; splitting it back "
                "into S_L and S_R is not a supported operation (memo 4.7)"
            )
        if not str(self.backend or "").strip():
            raise BasisResponseError("backend must be declared (who produced these matrices)")
        isc = np.asarray(self.isc_off, dtype=np.int64)
        if isc.ndim != 2 or isc.shape[1] != 3:
            raise BasisResponseError(f"isc_off must be (n_s, 3) integer offsets, got {isc.shape}")
        object.__setattr__(self, "isc_off", isc)
        expected = set(_REPRESENTATION_BLOCKS[self.representation])
        if set(self.blocks) != expected:
            raise BasisResponseError(
                f"representation {self.representation!r} stores {sorted(expected)}, "
                f"got {sorted(self.blocks)}"
            )
        shapes = {name: values.shape for name, values in self.blocks.items()}
        if len(set(shapes.values())) != 1:
            raise BasisResponseError(f"blocks disagree in shape: {shapes}")
        if next(iter(self.blocks.values())).shape[0] != isc.shape[0]:
            raise BasisResponseError(
                f"blocks carry {next(iter(self.blocks.values())).shape[0]} images but isc_off has "
                f"{isc.shape[0]}"
            )

    # -- shape / dtype -------------------------------------------------------
    @property
    def no_u(self) -> int:
        return int(next(iter(self.blocks.values())).shape[1])

    @property
    def dtype(self) -> str:
        return str(next(iter(self.blocks.values())).dtype)

    # -- provenance ----------------------------------------------------------
    @property
    def epc_backend_class(self) -> str:
        """``hybrid`` unless the overlap response and ``D_H`` share a backend."""
        backends = {self.backend, self.context.electronic_derivative_backend}
        return self.backend if len(backends) == 1 else EPC_BACKEND_CLASS_HYBRID

    def artifact_fields(self) -> dict[str, Any]:
        """Exactly the fields the ``basis_response`` DAG kind declares."""
        return {
            "formalism_id": self.context.formalism_id,
            "representation": self.representation,
            "backend": self.backend,
            "method": self.method,
            "perturbation_definition": self.context.definition,
            "q": list(self.context.q),
            "convention": self.conventions(),
            "dtype": self.dtype,
            "units": self.units,
        }

    def conventions(self) -> dict[str, Any]:
        """The q-space record of S29, identical to the one ``D_H`` declares.

        ``q``, the Fourier signs, the block orientation, the atomic phase and
        the ``k+q``/``-q`` rules all come from :mod:`epc_fourier`, so a response
        and the ``D_H`` it is paired with are compared by dict equality
        (:func:`epc_fourier.require_same_conventions`) rather than by trusting
        two independently written docstrings.
        """
        return {
            **convention_fields(self.context.q, q_aware_source=self.context.q_aware_source),
            "intra_atomic_included": bool(self.intra_atomic_included),
            "geometry_signature": self.context.geometry_signature,
            "basis_signature": self.context.basis_signature,
        }

    def signature_node(
        self, *, geometry: dict[str, Any], overlap: dict[str, Any], mode: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """The signed DAG node (``shared/artifact_signature.py``)."""
        return epc_artifact_node(
            "basis_response",
            self.artifact_fields(),
            {"geometry": geometry, "overlap": overlap, "mode": mode},
        )

    def provenance(self) -> dict[str, Any]:
        """Everything a manifest has to state about this response (no matrices)."""
        return {
            **self.artifact_fields(),
            "basis_response_backend": self.backend,
            "electronic_derivative_backend": self.context.electronic_derivative_backend,
            "epc_backend_class": self.epc_backend_class,
            "intra_atomic_included": bool(self.intra_atomic_included),
            "no_u": self.no_u,
            "n_s": int(self.isc_off.shape[0]),
            "source": dict(self.context.source),
            "diagnostics": dict(self.diagnostics),
        }

    # -- what the contraction consumes ---------------------------------------
    def at_k(self, k: Any = GAMMA) -> BasisResponse:
        """Bloch sum the blocks at fractional ``k`` and wrap them for the formalism."""
        summed = {
            name: bloch_sum(values, self.isc_off, k) for name, values in self.blocks.items()
        }
        if self.representation == REPRESENTATION_D_S:
            return basis_response_D_S(summed["D_S"], backend=self.backend)
        if self.representation == REPRESENTATION_COVARIANT:
            # memo 4.2: S_R = S*Gamma, S_L = Gamma^dag*S. Exact, not a reconstruction
            # from a sum — Gamma carries the antisymmetric part D_S cannot.
            s, gamma = summed["S"], summed["Gamma"]
            left, right = gamma.conj().T @ s, s @ gamma
        else:
            left, right = summed["S_left"], summed["S_right"]
        return basis_response_S_L_S_R(
            left,
            right,
            intra_atomic_included=self.intra_atomic_included,
            backend=self.backend,
        )


_REPRESENTATION_BLOCKS = {
    REPRESENTATION_S_L_S_R: ("S_left", "S_right"),
    REPRESENTATION_D_S: ("D_S",),
    REPRESENTATION_COVARIANT: ("Gamma", "S"),
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def from_S_L_S_R(
    S_left: Any,
    S_right: Any,
    context: ResponseContext,
    *,
    backend: str,
    method: str,
    intra_atomic_included: bool,
    isc_off: Any = ((0, 0, 0),),
    diagnostics: dict[str, Any] | None = None,
) -> BasisResponseArtifact:
    """The production representation, from any backend that produces both halves."""
    left = _blocks("S_left", S_left)
    right = _blocks("S_right", S_right, no_u=left.shape[1])
    return BasisResponseArtifact(
        representation=REPRESENTATION_S_L_S_R,
        blocks={"S_left": left, "S_right": right},
        isc_off=isc_off,
        backend=backend,
        method=method,
        intra_atomic_included=bool(intra_atomic_included),
        context=context,
        diagnostics=dict(diagnostics or {}),
    )


def from_covariant_connection(
    Gamma: Any,
    S: Any,
    context: ResponseContext,
    *,
    backend: str,
    method: str,
    intra_atomic_included: bool,
    isc_off: Any = ((0, 0, 0),),
    diagnostics: dict[str, Any] | None = None,
) -> BasisResponseArtifact:
    """``Gamma = S^-1 S_R`` plus the overlap it is defined against (memo 4.2)."""
    gamma = _blocks("Gamma", Gamma)
    overlap = _blocks("S", S, no_u=gamma.shape[1])
    return BasisResponseArtifact(
        representation=REPRESENTATION_COVARIANT,
        blocks={"Gamma": gamma, "S": overlap},
        isc_off=isc_off,
        backend=backend,
        method=method,
        intra_atomic_included=bool(intra_atomic_included),
        context=context,
        diagnostics=dict(diagnostics or {}),
    )


def from_D_S(
    D_S: Any,
    context: ResponseContext,
    *,
    backend: str,
    method: str,
    isc_off: Any = ((0, 0, 0),),
    diagnostics: dict[str, Any] | None = None,
) -> BasisResponseArtifact:
    """Sum-only response. Feeds the degraded symmetric formalism and nothing else."""
    return BasisResponseArtifact(
        representation=REPRESENTATION_D_S,
        blocks={"D_S": _blocks("D_S", D_S)},
        isc_off=isc_off,
        backend=backend,
        method=method,
        # A same-atom block contributes nothing to D_S, so there is no honest
        # sense in which this representation includes the intra-atomic term.
        intra_atomic_included=False,
        context=context,
        diagnostics=dict(diagnostics or {}),
    )


# ---------------------------------------------------------------------------
# SIESTA dS/dR -> (S_L, S_R), inter-atomic part
# ---------------------------------------------------------------------------


def inter_atomic_split(
    dS_by_atom_axis: Mapping[tuple[int, int], np.ndarray],
    direction: Any,
    atom_of_orbital: Any,
) -> dict[str, Any]:
    """Split atom-resolved ``dS/dR`` into ``S_L[u]`` and ``S_R[u]`` (memo section 6).

    ``dS_by_atom_axis`` maps ``(atom 1-based, axis 1|2|3)`` to ``(n_s, no_u, no_u)``
    blocks of ``dS/dR`` in ``1/Ang``; ``direction`` is the ``(N, 3)`` pattern.
    ``S_{mu,nu}`` moves with ``tau_atom(nu) - tau_atom(mu)``, so displacing atom
    ``I`` picks out ``<d phi_mu|phi_nu>`` on rows sitting on ``I`` and
    ``<phi_mu|d phi_nu>`` on columns sitting on ``I``. Same-atom blocks are
    invariant and drop out of both — that is the missing intra-atomic term, and
    it is reported, not silently zeroed.
    """
    atom_of_orbital = np.asarray(atom_of_orbital, dtype=np.int64).reshape(-1)
    vectors = np.asarray(direction, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise BasisResponseError(f"direction must be (N, 3), got shape {vectors.shape}")
    if not dS_by_atom_axis:
        raise BasisResponseError("no dS/dR records given")

    row_atom = atom_of_orbital[:, None]
    col_atom = atom_of_orbital[None, :]
    intra = row_atom == col_atom

    shapes = {np.asarray(values).shape for values in dS_by_atom_axis.values()}
    if len(shapes) != 1:
        raise BasisResponseError(f"dS/dR records disagree in shape: {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) != 3 or shape[1] != atom_of_orbital.size or shape[2] != atom_of_orbital.size:
        raise BasisResponseError(
            f"dS/dR blocks must be (n_s, no_u, no_u) with no_u = {atom_of_orbital.size}, got {shape}"
        )

    # ponytail: dense (n_s, no_u, no_u). Fine to graphene/AB scale; MATBG needs
    # the same masks applied to the sparse (row, col, image) arrays instead.
    left = np.zeros(shape, dtype=np.float64)
    right = np.zeros(shape, dtype=np.float64)
    intra_residual = 0.0
    used: list[tuple[int, int]] = []

    for atom in range(1, vectors.shape[0] + 1):
        for axis in (1, 2, 3):
            weight = float(vectors[atom - 1, axis - 1])
            if weight == 0.0:
                continue
            values = dS_by_atom_axis.get((int(atom), int(axis)))
            if values is None:
                raise BasisResponseError(
                    f"direction displaces atom {atom} along axis {axis} (weight {weight:+.6g}) but "
                    f"no dS/dR record was given for it; available: {sorted(dS_by_atom_axis)}"
                )
            values = np.asarray(values, dtype=np.float64)
            on_atom_row = (row_atom == atom) & ~intra
            on_atom_col = (col_atom == atom) & ~intra
            left += weight * values * on_atom_row
            right += weight * values * on_atom_col
            intra_residual = max(intra_residual, float(np.abs(values * intra).max(initial=0.0)))
            used.append((int(atom), int(axis)))

    return {
        "S_left": left,
        "S_right": right,
        "intra_atomic_residual": intra_residual,
        "records_used": used,
    }


def from_dhsdr(
    dhsdr: Any,
    context: ResponseContext,
    *,
    atom_of_orbital: Any,
    spin: int = 0,
    backend: str = "siesta",
) -> BasisResponseArtifact:
    """Build the response from a ``*.dHSdR.nc`` read by ``read_siesta_dhsdr``.

    Inter-atomic only: the artifact declares ``intra_atomic_included=False`` and
    says so in ``method``, so no consumer can mistake it for the complete
    response the production formalism eventually needs (memo A2, closed by the
    analytic monocentric integral, not by this backend).
    """
    if context.direction is None:
        raise BasisResponseError("from_dhsdr needs a Direction: the split is contracted with it")
    vectors = np.asarray(context.direction.vectors, dtype=np.float64)

    records: dict[tuple[int, int], np.ndarray] = {}
    isc_off = None
    for atom in range(1, vectors.shape[0] + 1):
        for axis in (1, 2, 3):
            if vectors[atom - 1, axis - 1] == 0.0:
                continue
            record = dhsdr.derivative(atom, axis, "D_S")
            records[(atom, axis)] = record.to_dense(spin=spin)
            isc_off = record.isc_off if isc_off is None else isc_off
            if not np.array_equal(np.asarray(isc_off), np.asarray(record.isc_off)):
                raise BasisResponseError(
                    f"image table of atom {atom} axis {axis} differs from the first record; "
                    "the blocks cannot be added"
                )

    split = inter_atomic_split(records, vectors, atom_of_orbital)
    if split["intra_atomic_residual"] > INTRA_ATOMIC_RESIDUAL_TOL:
        raise BasisResponseError(
            f"dS/dR carries {split['intra_atomic_residual']:.3e} 1/Ang on same-atom blocks, which "
            f"must vanish identically (memo section 6). The orbital -> atom map or the sign "
            f"convention is wrong; the split would silently mix S_L and S_R"
        )

    source = {
        **dict(context.source),
        "dhsdr": dhsdr.metadata() if hasattr(dhsdr, "metadata") else None,
        "spin": int(spin),
    }
    return from_S_L_S_R(
        split["S_left"],
        split["S_right"],
        ResponseContext(
            geometry_signature=context.geometry_signature,
            basis_signature=context.basis_signature,
            electronic_derivative_backend=context.electronic_derivative_backend,
            direction=context.direction,
            q=context.q,
            q_aware_source=context.q_aware_source,
            source=source,
            formalism_id=context.formalism_id,
        ),
        backend=backend,
        method=METHOD_DHSDR_INTER_ATOMIC,
        intra_atomic_included=False,
        isc_off=isc_off if isc_off is not None else ((0, 0, 0),),
        diagnostics={
            "intra_atomic_residual_inv_ang": split["intra_atomic_residual"],
            "records_used": split["records_used"],
            "intra_atomic_term": "omitted; monocentric <d phi_mu|phi_nu> integral not implemented "
            "(memo section 6, assumption A2)",
        },
    )


# ---------------------------------------------------------------------------
# C14C: validation
# ---------------------------------------------------------------------------

VALIDATION_SCHEMA = "epc_basis_response_validation_v1"

VERDICT_PASS = "PASS"
VERDICT_NO_GO = "NO_GO"

# Pre-registered before any measured response was looked at. They are not
# accuracy targets: every one of them gates an *exact* algebraic identity, so
# the only thing they have to absorb is float64 assembly noise. The one place
# where a real, measured floor enters is ``tau_reference``, which the caller
# takes from the GO-2 certification of the very D_S it compares against.
TOLERANCES: dict[str, float] = {
    # ||S_L + S_R - D_S_ref|| <= max(tau_reference, sum_relative * ||D_S_ref||)
    "sum_relative": 1e-9,
    # ||S_L(k) - S_R(k)^dag|| <= max(tau_reference, identity_relative * ||S_L(k)||),
    # memo section 3
    "identity_relative": 1e-8,
    # rigid translation: ||S_L + S_R|| <= max(tau_reference, translation_relative * ||S_L||)
    "translation_relative": 1e-6,
    # g invariance under H -> H + cS, and g -> U^dag g U inside a degenerate block
    "gauge_relative": 1e-9,
}

# Two generic k plus Gamma: at a generic k every image of the table carries a
# different phase, so a wrong image partner cannot cancel out of the identity.
DEFAULT_K_POINTS = (GAMMA, (0.25, 0.0, 0.0), (0.13, 0.41, -0.2))

# fd_perturbation_space.KIND_UNIFORM_TRANSLATION, duplicated as a string so this
# module does not pull scipy in through that import.
KIND_UNIFORM_TRANSLATION = "uniform_translation"

# The symmetric form's error is (eps_m - eps_n) A, so over a whole valence+
# conduction block it is dominated by pairs tens of eV apart, which no phonon
# connects. On shell |eps_m - eps_n| = hbar*omega, and for graphene the highest
# mode is ~0.196 eV (memo section 8), so this window is where the number that
# matters lives. It is a diagnostic window, not a claim about any mode.
ON_SHELL_WINDOW_EV = 0.25

# Assumption A2 of the memo: the same-atom blocks of A. Not a Pulay/incompleteness
# term (that one is A1 and the memo closes it structurally, section 4.3) but an
# unimplemented integral, and therefore a hard gate rather than an error bar.
INTRA_ATOMIC_TERM = {
    "term": "intra_atomic_basis_response",
    "assumption": "A2 (docs/epc_formalismo_pao_movil.md sections 6 and 10)",
    "lives_in": "same-atom blocks of S_L and S_R, all images; identically zero in dS/dR",
    "resolution": "analytic monocentric <d_alpha phi_mu | phi_nu> over the .ion.xml radials",
    "policy": "NO_GO while unresolved; no correction factor is derived from its sensitivity",
}

OUT_OF_SPAN_TERM = {
    "term": "out_of_span_projection",
    "assumption": "A1 (docs/epc_formalismo_pao_movil.md section 4.3)",
    "lives_in": "Pi = <d phi|(1 - P) H|phi> + h.c.",
    "resolution": "structural: it is the definition of the PAO model, and it is what makes the "
    "diagonal of (F2) reproduce d(eps_n) exactly (generalized Hellmann-Feynman)",
    "policy": "closed_non_blocking; verified by the HF diagonal gate, not by a factor",
}


def _check(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail, "applicable": True, **extra}


def _norm(values: Any) -> float:
    return float(np.linalg.norm(np.asarray(values)))


def _as_square(name: str, array: Any) -> np.ndarray:
    matrix = np.asarray(array)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise BasisResponseError(f"{name} must be a square matrix, got shape {matrix.shape}")
    return matrix


def _relative(value: float, scale: float) -> float | None:
    return float(value / scale) if scale > 0.0 else None


def _split_norms(values: Any) -> dict[str, Any]:
    """Frobenius norms of a coupling block, split into diagonal and interband."""
    matrix = np.asarray(values)
    total = _norm(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        # k+q != k: there is no band-diagonal to speak of.
        return {"frobenius": total, "diagonal_frobenius": None, "interband_frobenius": total}
    diagonal = np.diag(matrix)
    return {
        "frobenius": total,
        "diagonal_frobenius": _norm(diagonal),
        "interband_frobenius": _norm(matrix - np.diag(diagonal)),
    }


def real_space_total(artifact: BasisResponseArtifact) -> np.ndarray | None:
    """``D_S`` blocks implied by the artifact, or ``None`` if only defined at ``k``.

    ``covariant_basis_connection`` stores ``Gamma`` and ``S``, whose product is a
    convolution over images: it has no blockwise real-space total, and this
    returns ``None`` rather than pretending otherwise.
    """
    if artifact.representation == REPRESENTATION_S_L_S_R:
        return artifact.blocks["S_left"] + artifact.blocks["S_right"]
    if artifact.representation == REPRESENTATION_D_S:
        return artifact.blocks["D_S"]
    return None


def check_sum_against_reference(
    artifact: BasisResponseArtifact,
    D_S_reference: Any,
    *,
    reference_name: str,
    tau_reference: float = 0.0,
    relative_tol: float = TOLERANCES["sum_relative"],
) -> dict[str, Any]:
    """``S_L + S_R`` against an independently measured ``D_S`` (criterion 1).

    ``tau_reference`` is the reference's own noise floor — GO-2's measured
    ``tau_FD`` when the reference is a central finite difference — so a
    discrepancy below it is indistinguishable from the reference's resolution.
    Passing bounds the *hermitian* part of the response only; ``A`` is invisible
    here by construction (memo 4.7), which is why this is one check of several.
    """
    total = real_space_total(artifact)
    if total is None:
        return {
            "check": "sum_reproduces_reference",
            "passed": False,
            "applicable": False,
            "detail": f"representation {artifact.representation!r} has no real-space D_S; "
            "cross-check it at k against a Bloch-summed reference instead",
        }
    reference = _blocks("D_S_reference", D_S_reference, no_u=artifact.no_u)
    if reference.shape != total.shape:
        raise BasisResponseError(
            f"D_S reference has shape {reference.shape}, the response {total.shape}; they are not "
            "on the same image table"
        )
    residual = _norm(total - reference)
    scale = _norm(reference)
    tolerance = max(float(tau_reference), relative_tol * scale)
    return _check(
        "sum_reproduces_reference",
        residual <= tolerance,
        f"||S_L + S_R - D_S({reference_name})|| = {residual:.6e} vs tolerance {tolerance:.6e} "
        f"= max(tau_reference {float(tau_reference):.6e}, {relative_tol:g} * {scale:.6e})",
        reference=reference_name,
        residual_frobenius=residual,
        reference_frobenius=scale,
        relative_residual=_relative(residual, scale),
        tau_reference=float(tau_reference),
        tolerance=tolerance,
        bounds="hermitian part only; A is invisible to any D_S reference (memo 4.7)",
    )


def check_split_identity(
    artifact: BasisResponseArtifact,
    *,
    k_points: Sequence[Any] = DEFAULT_K_POINTS,
    tau_reference: float = 0.0,
    relative_tol: float = TOLERANCES["identity_relative"],
) -> dict[str, Any]:
    """``S_L(k) = S_R(k)^dag`` at every ``k`` — exact algebra, not an approximation.

    This is what separates a real split from a sum cut in half: a swapped
    row/column mask or a wrong image partner survives
    :func:`check_sum_against_reference` and dies here.

    The identity is exact algebra, but a response assembled from a file inherits
    that file's resolution: ``tau_reference`` lets the caller state the measured
    floor (GO-2's ``tau_FD`` for a SIESTA-derived response) below which nothing
    can be called a discrepancy.
    """
    worst = 0.0
    scale = 0.0
    by_k = []
    for k in k_points:
        response = artifact.at_k(k)
        if response.S_left is None or response.S_right is None:
            return {
                "check": "split_identity_S_L_equals_S_R_dagger",
                "passed": False,
                "applicable": False,
                "detail": f"representation {artifact.representation!r} carries no split to test",
            }
        residual = _norm(response.S_left - response.S_right.conj().T)
        magnitude = _norm(response.S_left)
        worst = max(worst, residual)
        scale = max(scale, magnitude)
        by_k.append(
            {
                "k": [float(value) for value in np.asarray(k, dtype=np.float64).reshape(3)],
                "residual_frobenius": residual,
                "S_left_frobenius": magnitude,
                "antisymmetric_frobenius": _norm(response.antisymmetric),
            }
        )
    tolerance = max(float(tau_reference), relative_tol * scale)
    return _check(
        "split_identity_S_L_equals_S_R_dagger",
        worst <= tolerance,
        f"max_k ||S_L - S_R^dag|| = {worst:.6e} vs tolerance {tolerance:.6e} "
        f"= max(tau_reference {float(tau_reference):.6e}, {relative_tol:g} * {scale:.6e})",
        max_residual_frobenius=worst,
        tau_reference=float(tau_reference),
        tolerance=tolerance,
        by_k=by_k,
    )


def check_uniform_translation(
    artifact: BasisResponseArtifact,
    *,
    tau_reference: float = 0.0,
    relative_tol: float = TOLERANCES["translation_relative"],
) -> dict[str, Any]:
    """Rigid translation: ``S_L + S_R`` must vanish while ``S_L`` must not.

    Moving every atom alike leaves ``S`` invariant, so ``D_S = 0`` identically —
    and a response that also reported ``S_L = 0`` would be reporting that the
    orbitals do not move with their centres. The measured ``||A||`` here is the
    inter-atomic antisymmetric response of the acoustic mode.
    """
    direction = artifact.context.direction
    kind = str(getattr(direction, "kind", "") or "")
    if kind != KIND_UNIFORM_TRANSLATION:
        return {
            "check": "uniform_translation_sum_vanishes",
            "passed": True,
            "applicable": False,
            "detail": f"direction kind is {kind or 'unset'!r}, not {KIND_UNIFORM_TRANSLATION!r}",
        }
    if artifact.representation != REPRESENTATION_S_L_S_R:
        return {
            "check": "uniform_translation_sum_vanishes",
            "passed": False,
            "applicable": False,
            "detail": f"representation {artifact.representation!r} carries no S_L to weigh the "
            "vanishing sum against; the acoustic statement is about the split, not about D_S",
        }
    total = real_space_total(artifact)
    residual = _norm(total)
    scale = _norm(artifact.blocks["S_left"])
    tolerance = max(float(tau_reference), relative_tol * scale)
    return _check(
        "uniform_translation_sum_vanishes",
        residual <= tolerance and scale > 0.0,
        f"||S_L + S_R|| = {residual:.6e} vs tolerance {tolerance:.6e}, with ||S_L|| = {scale:.6e} "
        f"(the response of the acoustic mode is antisymmetric, not zero)",
        residual_frobenius=residual,
        S_left_frobenius=scale,
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class ElectronicState:
    """One ``k``: the electrons the response is contracted between.

    ``C`` is ``S(k)``-orthonormal and ``eps`` are the eigenvalues of the same
    pencil (C15). ``D_H`` is the directional derivative of ``H`` at that ``k``,
    in ``eV/Ang``, for the *same* displacement pattern as the response.
    """

    label: str
    k: tuple[float, float, float]
    D_H: np.ndarray
    C: np.ndarray
    eps: np.ndarray

    def __post_init__(self) -> None:
        k = tuple(float(value) for value in np.asarray(self.k, dtype=np.float64).reshape(-1))
        if len(k) != 3:
            raise BasisResponseError(f"k must be a fractional 3-vector, got {self.k!r}")
        object.__setattr__(self, "k", k)
        d_h = np.asarray(self.D_H)
        c = np.asarray(self.C)
        eps = np.asarray(self.eps, dtype=np.float64).reshape(-1)
        if d_h.ndim != 2 or d_h.shape[0] != d_h.shape[1]:
            raise BasisResponseError(f"D_H must be square, got {d_h.shape}")
        if c.ndim != 2 or c.shape[0] != d_h.shape[0]:
            raise BasisResponseError(f"C must be (no_u, n_bands) with no_u = {d_h.shape[0]}, got {c.shape}")
        if eps.size != c.shape[1]:
            raise BasisResponseError(f"eps carries {eps.size} values for {c.shape[1]} bands")
        object.__setattr__(self, "D_H", d_h)
        object.__setattr__(self, "C", c)
        object.__setattr__(self, "eps", eps)


def degenerate_groups(eps: Any, tol_ev: float = 1e-6) -> list[list[int]]:
    """Indices of the (near-)degenerate groups of ``eps``, groups of one dropped."""
    values = np.asarray(eps, dtype=np.float64).reshape(-1)
    groups: list[list[int]] = []
    current = [0] if values.size else []
    for index in range(1, values.size):
        if abs(values[index] - values[index - 1]) <= tol_ev:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    if current:
        groups.append(current)
    return [group for group in groups if len(group) > 1]


def contribution_split(
    artifact: BasisResponseArtifact,
    state: ElectronicState,
    *,
    degeneracy_tol_ev: float = 1e-6,
    on_shell_window_ev: float = ON_SHELL_WINDOW_EV,
) -> dict[str, Any]:
    """How much of ``g`` is the basis response, and what each approximation drops.

    Three contractions of the same ``D_H`` and the same electrons:

    * production — (F2), the response's own representation;
    * ``moving_pao_symmetric_overlap_v1`` — the same response reduced to its sum,
      i.e. ``A`` dropped. The difference is exactly ``(eps_m - eps_n) Cm^dag A Cn``:
      zero on the diagonal and inside degenerate blocks by construction, so
      whatever this reports off-diagonal is the size of the term the roadmap
      refused to adopt by intuition;
    * ``moving_pao_ignore_overlap_v1`` — ``C^dag D_H C``, the control negative.
      Its delta *is* the basis response's contribution to ``g``.
    """
    response = artifact.at_k(state.k)
    kwargs = {"C_row": state.C, "C_col": state.C, "eps_row": state.eps, "eps_col": state.eps}
    production = epc_matrix_elements(state.D_H, response, **kwargs)
    control = epc_matrix_elements(state.D_H, None, control_negative=True, **kwargs)
    if response.representation == REPRESENTATION_S_L_S_R:
        symmetric = epc_matrix_elements(
            state.D_H, basis_response_D_S(response.total, backend=artifact.backend), **kwargs
        )
    else:
        symmetric = production  # a D_S response already *is* the symmetric form

    reference = _split_norms(production.values)

    def delta(other: Any) -> dict[str, Any]:
        difference = _split_norms(np.asarray(other.values) - np.asarray(production.values))
        return {
            **difference,
            "relative_frobenius": _relative(difference["frobenius"], reference["frobenius"]),
            "relative_diagonal": _relative(
                difference["diagonal_frobenius"] or 0.0, reference["diagonal_frobenius"] or 0.0
            ),
            "relative_interband": _relative(
                difference["interband_frobenius"], reference["interband_frobenius"]
            ),
        }

    ignored = delta(control)

    def on_shell(other: Any) -> dict[str, Any]:
        """The same error restricted to the band pairs a phonon can connect."""
        if np.asarray(other.values).shape != np.asarray(production.values).shape:
            return {"applicable": False, "reason": "row and column states differ (k+q != k)"}
        gaps = np.abs(state.eps[:, None] - state.eps[None, :])
        mask = (gaps <= float(on_shell_window_ev)) & ~np.eye(state.eps.size, dtype=bool)
        difference = (np.asarray(other.values) - np.asarray(production.values)) * mask
        reference_block = np.asarray(production.values) * mask
        return {
            "applicable": True,
            "window_ev": float(on_shell_window_ev),
            "pair_count": int(mask.sum()),
            "frobenius": _norm(difference),
            "g_frobenius": _norm(reference_block),
            "relative": _relative(_norm(difference), _norm(reference_block)),
        }

    return {
        "state": state.label,
        "k": list(state.k),
        "band_count": int(state.eps.size),
        "degenerate_groups": degenerate_groups(state.eps, degeneracy_tol_ev),
        "units": "eV/Ang per unit-Frobenius displacement",
        "production": {"formalism_id": production.formalism_id, **reference},
        # What the overlap/basis response itself contributes: production minus the
        # control negative, split the way the ticket asks for it.
        "basis_response_contribution": ignored,
        "approximations": {
            FORMALISM_ID_SYMMETRIC: {
                "status": "approximation",
                "drops": DEGRADED_FORMALISM_IDS[FORMALISM_ID_SYMMETRIC],
                "exact_where": "diagonal, degenerate blocks, adiabatic limit",
                # The whole-block norm is dominated by pairs no phonon connects;
                # this is the same error where it is actually consumed.
                "on_shell": on_shell(symmetric),
                **delta(symmetric),
            },
            FORMALISM_ID_IGNORE_OVERLAP: {
                "status": "approximation",
                "drops": DEGRADED_FORMALISM_IDS[FORMALISM_ID_IGNORE_OVERLAP],
                "exact_where": "nowhere; control negative only",
                **ignored,
            },
        },
    }


def reference_discrepancy_in_subspace(
    artifact: BasisResponseArtifact, state: ElectronicState, D_S_at_k: Any
) -> dict[str, Any]:
    """The reference's own resolution, propagated into the electronic subspace.

    ``check_sum_against_reference`` gates the matrices; this says what that same
    disagreement is worth where it is actually consumed, following the roadmap's
    prescription ``delta_g = C_f^dag delta_Delta C_i`` instead of converting a
    global matrix norm into a percentage of ``g``. It is a measured size, not a
    gate — nothing here passes or fails.
    """
    difference = _as_square("D_S_at_k", D_S_at_k) - artifact.at_k(state.k).total
    projected = state.C.conj().T @ difference @ state.C
    weights = 0.5 * (state.eps[:, None] + state.eps[None, :])
    return {
        "state": state.label,
        "matrix_frobenius": _norm(difference),
        "subspace_frobenius": _norm(projected),
        "induced_g_frobenius": _norm(weights * projected),
        "units": "1/Ang for the matrices, eV/Ang for the induced g",
        "meaning": "how much of g the difference between the response's sum and the "
        "independent D_S reference could account for; a resolution, not an error bar on g",
    }


def check_energy_origin_invariance(
    artifact: BasisResponseArtifact,
    state: ElectronicState,
    *,
    D_S_at_k: Any = None,
    shift_ev: float = 1.0,
    relative_tol: float = TOLERANCES["gauge_relative"],
) -> dict[str, Any]:
    """``H -> H + cS`` must leave ``g`` alone (memo section 9).

    Under the shift ``D_H -> D_H + c D_S`` and ``eps -> eps + c``; the induced
    terms cancel *because* ``D_S = S_L + S_R``. Pass ``D_S_at_k`` from the
    independent reference to make this a statement about the response: with the
    response's own sum it is an identity and can only catch a broken (F2).
    """
    response = artifact.at_k(state.k)
    total = response.total if D_S_at_k is None else _as_square("D_S_at_k", D_S_at_k)
    kwargs = {"C_row": state.C, "C_col": state.C}
    base = epc_matrix_elements(
        state.D_H, response, eps_row=state.eps, eps_col=state.eps, **kwargs
    )
    shifted = epc_matrix_elements(
        state.D_H + float(shift_ev) * total,
        response,
        eps_row=state.eps + float(shift_ev),
        eps_col=state.eps + float(shift_ev),
        **kwargs,
    )
    residual = _norm(np.asarray(shifted.values) - np.asarray(base.values))
    scale = _norm(base.values)
    tolerance = relative_tol * scale
    return _check(
        "energy_origin_invariance",
        residual <= tolerance,
        f"||g(H + {shift_ev:g} S) - g(H)|| = {residual:.6e} vs tolerance {tolerance:.6e}",
        state=state.label,
        D_S_source="independent_reference" if D_S_at_k is not None else "response_own_sum",
        residual_frobenius=residual,
        g_frobenius=scale,
        tolerance=tolerance,
    )


def check_gauge_covariance(
    artifact: BasisResponseArtifact,
    state: ElectronicState,
    *,
    degeneracy_tol_ev: float = 1e-6,
    seed: int = 0,
    relative_tol: float = TOLERANCES["gauge_relative"],
) -> dict[str, Any]:
    """``C -> C U`` inside a degenerate block must give ``g -> U^dag g U``.

    Band-indexed elements are gauge; the singular values are not, and those are
    what may be published. Not applicable when the spectrum has no degeneracy at
    this ``k`` — which is stated, not silently passed.

    Covariance is exact only for an exactly degenerate block. A real DFT
    spectrum is degenerate to the level its grid breaks the symmetry at, and the
    residual then inherits that: the tolerance carries ``spread * ||C^dag S C||``
    on top of the algebraic floor, computed from the block being rotated rather
    than chosen to fit.
    """
    groups = degenerate_groups(state.eps, degeneracy_tol_ev)
    if not groups:
        return {
            "check": "degenerate_gauge_covariance",
            "passed": True,
            "applicable": False,
            "detail": f"no degenerate group within {degeneracy_tol_ev:g} eV at {state.label}",
            "state": state.label,
        }
    rotation = random_gauge_rotation(groups, state.eps.size, np.random.default_rng(seed))

    response = artifact.at_k(state.k)
    kwargs = {"eps_row": state.eps, "eps_col": state.eps}
    base = epc_matrix_elements(state.D_H, response, C_row=state.C, C_col=state.C, **kwargs)
    rotated = epc_matrix_elements(
        state.D_H, response, C_row=state.C @ rotation, C_col=state.C @ rotation, **kwargs
    )
    expected = rotation.conj().T @ np.asarray(base.values) @ rotation
    residual = _norm(np.asarray(rotated.values) - expected)
    singular_shift = _norm(
        np.linalg.svd(np.asarray(rotated.values), compute_uv=False)
        - np.linalg.svd(np.asarray(base.values), compute_uv=False)
    )
    scale = _norm(base.values)
    # What the rotation cannot commute with: eps inside the block is only
    # constant up to its own spread, and it multiplies the overlap terms of (F2).
    spread = max(float(state.eps[group].max() - state.eps[group].min()) for group in groups)
    if response.representation == REPRESENTATION_S_L_S_R:
        mixing = _norm(state.C.conj().T @ response.S_left @ state.C) + _norm(
            state.C.conj().T @ response.S_right @ state.C
        )
    else:
        mixing = _norm(state.C.conj().T @ response.total @ state.C)
    tolerance = relative_tol * scale + spread * mixing
    return _check(
        "degenerate_gauge_covariance",
        residual <= tolerance and singular_shift <= tolerance,
        f"||g(CU) - U^dag g U|| = {residual:.6e} and singular-value shift {singular_shift:.6e} "
        f"vs tolerance {tolerance:.6e} = {relative_tol:g} * {scale:.6e} + "
        f"{spread:.3e} eV * {mixing:.6e}",
        state=state.label,
        degeneracy_tol_ev=float(degeneracy_tol_ev),
        degenerate_groups=groups,
        degeneracy_spread_ev=spread,
        covariance_residual=residual,
        singular_value_shift=singular_shift,
        tolerance=tolerance,
    )


def intra_atomic_sensitivity(state: ElectronicState, atom_of_orbital: Any) -> dict[str, Any]:
    """How far ``g`` moves per unit of the *unmeasured* intra-atomic ``A``.

    For each same-atom orbital pair, a unit-Frobenius anti-hermitian
    ``E = (e_mu e_nu^dag - e_nu e_mu^dag)/sqrt(2)`` is pushed through the ``A``
    term of (F3), ``(eps_m - eps_n) Cm^dag E Cn``, and the induced interband norm
    is recorded. This is a derivative of ``g`` with respect to a term nobody has
    computed yet — it says how accurate the monocentric integral must be, and it
    is explicitly *not* a value of that term nor a factor to correct anything by.
    """
    atoms = np.asarray(atom_of_orbital, dtype=np.int64).reshape(-1)
    if atoms.size != state.C.shape[0]:
        raise BasisResponseError(
            f"atom_of_orbital carries {atoms.size} entries for {state.C.shape[0]} orbitals"
        )
    gaps = state.eps[:, None] - state.eps[None, :]
    conjugated = state.C.conj()  # (C^dag e_mu)_m = conj(C[mu, m])
    worst = 0.0
    total = 0.0
    pairs = 0
    worst_pair: tuple[int, int] | None = None
    for mu in range(atoms.size):
        for nu in range(mu + 1, atoms.size):
            if atoms[mu] != atoms[nu]:
                continue
            x = conjugated[mu, :]
            y = conjugated[nu, :]
            block = (np.outer(x, y.conj()) - np.outer(y, x.conj())) / np.sqrt(2.0)
            magnitude = _norm(gaps * block)
            total += magnitude**2
            pairs += 1
            if magnitude > worst:
                worst, worst_pair = magnitude, (mu, nu)
    return {
        "state": state.label,
        "same_atom_orbital_pairs": pairs,
        "max_interband_response": worst,
        "rms_interband_response": float(np.sqrt(total / pairs)) if pairs else 0.0,
        "max_pair_orbitals_zero_based": list(worst_pair) if worst_pair else None,
        "units": "eV/Ang of g per unit-Frobenius 1/Ang of intra-atomic A",
        "interpretation": "sensitivity of g to the omitted term, not its value; "
        "no correction factor is derived from it",
    }


def omitted_terms(artifact: BasisResponseArtifact) -> list[dict[str, Any]]:
    """Every term the artifact does not carry, with its status."""
    terms = [dict(OUT_OF_SPAN_TERM, status="closed_non_blocking")]
    if artifact.intra_atomic_included:
        terms.append(dict(INTRA_ATOMIC_TERM, status="included_by_backend"))
    else:
        terms.append(
            dict(
                INTRA_ATOMIC_TERM,
                status="unresolved",
                declared_by=f"intra_atomic_included=False, method={artifact.method!r}",
            )
        )
    if artifact.representation == REPRESENTATION_D_S:
        terms.append(
            {
                "term": "antisymmetric_overlap_response_A",
                "assumption": "memo 4.2 / 4.7",
                "lives_in": "A = (S_L - S_R)/2",
                "resolution": "supply S_L and S_R separately; A cannot be recovered from a sum",
                "policy": "the artifact is restricted to the degraded symmetric formalism",
                "status": "unresolved",
            }
        )
    return terms


def validate_basis_response(
    artifact: BasisResponseArtifact,
    *,
    D_S_reference: Any = None,
    reference_name: str = "",
    tau_reference: float = 0.0,
    k_points: Sequence[Any] = DEFAULT_K_POINTS,
    states: Sequence[ElectronicState] = (),
    atom_of_orbital: Any = None,
    degeneracy_tol_ev: float = 1e-6,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The C14C gate: does this response reproduce its reference, and what is missing.

    ``D_S_reference`` is real-space blocks on the artifact's own image table,
    measured independently of the artifact (a central finite difference of ``S``,
    or an analytic overlap derivative); ``tau_reference`` is that reference's own
    noise floor, GO-2's ``tau_FD`` when it comes from finite differences.
    ``states`` are the ``k`` at which the contribution to ``g`` is quantified —
    without them the report carries the identities but no magnitudes, and says so.

    The verdict is ``PASS`` only when every applicable check passes, the
    reference comparison was actually evaluated, the formalism is not a degraded
    variant, and no omitted term is left ``unresolved``. An unresolved term is
    never converted into a correction factor.
    """
    limits = {**TOLERANCES, **dict(tolerances or {})}
    checks = [
        check_split_identity(
            artifact,
            k_points=k_points,
            tau_reference=tau_reference,
            relative_tol=limits["identity_relative"],
        ),
        check_uniform_translation(
            artifact, tau_reference=tau_reference, relative_tol=limits["translation_relative"]
        ),
    ]
    if D_S_reference is None:
        checks.insert(
            0,
            {
                "check": "sum_reproduces_reference",
                "passed": False,
                "applicable": False,
                "detail": "no independent D_S reference was supplied; acceptance criterion 1 "
                "cannot be evaluated for this artifact",
            },
        )
    else:
        checks.insert(
            0,
            check_sum_against_reference(
                artifact,
                D_S_reference,
                reference_name=reference_name or "unnamed_reference",
                tau_reference=tau_reference,
                relative_tol=limits["sum_relative"],
            ),
        )

    contributions = []
    for state in states:
        contribution = contribution_split(artifact, state, degeneracy_tol_ev=degeneracy_tol_ev)
        if D_S_reference is not None:
            # The reference is a *measurement*: its disagreement with the response
            # is gated on the matrices (above) and only reported here, propagated
            # into the subspace where g lives.
            contribution["reference_discrepancy_in_subspace"] = reference_discrepancy_in_subspace(
                artifact,
                state,
                bloch_sum(_blocks("D_S_reference", D_S_reference), artifact.isc_off, state.k),
            )
        contributions.append(contribution)
        checks.append(
            check_energy_origin_invariance(artifact, state, relative_tol=limits["gauge_relative"])
        )
        checks.append(
            check_gauge_covariance(
                artifact,
                state,
                degeneracy_tol_ev=degeneracy_tol_ev,
                relative_tol=limits["gauge_relative"],
            )
        )

    terms = omitted_terms(artifact)
    if atom_of_orbital is not None and not artifact.intra_atomic_included:
        for term in terms:
            if term["term"] == INTRA_ATOMIC_TERM["term"]:
                term["sensitivity"] = [
                    intra_atomic_sensitivity(state, atom_of_orbital) for state in states
                ]

    reasons = []
    if any(check["applicable"] and not check["passed"] for check in checks):
        reasons.append(
            "an applicable check failed: "
            + ", ".join(
                check["check"] for check in checks if check["applicable"] and not check["passed"]
            )
        )
    if not next(check for check in checks if check["check"] == "sum_reproduces_reference")["applicable"]:
        reasons.append("the response was never compared against an independent D_S reference")
    if artifact.context.formalism_id in DEGRADED_FORMALISM_IDS:
        reasons.append(
            f"{artifact.context.formalism_id!r} is a degraded variant: diagnostic only, never truth"
        )
    unresolved = [term["term"] for term in terms if term["status"] == "unresolved"]
    if unresolved:
        reasons.append("unresolved term(s), no correction factor applied: " + ", ".join(unresolved))
    if not states:
        reasons.append("no electronic state was given: the contribution to g was not quantified")

    return {
        "schema": VALIDATION_SCHEMA,
        "ticket": "C14C / E-F_001-S17",
        "memo": "docs/epc_formalismo_pao_movil.md",
        "provenance": artifact.provenance(),
        "tolerances": limits,
        "tolerances_pre_registered": not dict(tolerances or {}),
        "checks": checks,
        "contributions": contributions,
        "omitted_terms": terms,
        "verdict": VERDICT_NO_GO if reasons else VERDICT_PASS,
        "reasons": reasons,
        "verdict_policy": "PASS requires every applicable check to pass, an evaluated reference "
        "comparison, a non-degraded formalism and no unresolved term",
    }


def basis_response_contract() -> dict[str, Any]:
    """The C14B contract as data, for manifests and the UI (no physics here)."""
    return {
        "schema": "epc_basis_response_contract_v1",
        "memo": "docs/epc_formalismo_pao_movil.md",
        "representations": list(BASIS_RESPONSE_REPRESENTATIONS),
        "required_by_formalism": {key: list(value) for key, value in REQUIRED_REPRESENTATIONS.items()},
        "units": UNITS,
        "conventions": convention_fields(),
        "convention_id": CONVENTION_ID,
        "sources": {
            "analytic_pao_overlap_derivative": "native S_L/S_R; production candidate, not implemented",
            "siesta_dhsdr": METHOD_DHSDR_INTER_ATOMIC,
            "covariant_connection": "Gamma = S^-1 S_R with S; equivalent to the pair",
            "frozen_overlap_fd": "D_S only; degraded",
        },
        "forbidden": "reconstructing S_L/S_R from D_S (memo 4.7)",
        "validation": {
            "schema": VALIDATION_SCHEMA,
            "entry_point": "validate_basis_response",
            "tolerances": dict(TOLERANCES),
            "verdict_values": [VERDICT_PASS, VERDICT_NO_GO],
            "unresolved_term_policy": INTRA_ATOMIC_TERM["policy"],
        },
    }
