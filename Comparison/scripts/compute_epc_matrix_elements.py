#!/usr/bin/env python3
"""C19 / E-F_001-S24: the PAO-covariant EPC contraction.

This is where the pieces stop being separate artifacts and become ``g``. The
chain is fixed by ``docs/epc_formalismo_pao_movil.md`` and nothing here may
short-circuit it:

    raw D_H[v]  +  basis response of the *same* v   ->  PAO-covariant response   (F1)
    PAO-covariant response  +  S-orthonormal C, eps             ->  g_mn         (F2)
    g_mn        x  the mode's own displacement      ->  g in eV

Three refusals are structural rather than advisory:

* **``D_H`` is never ``PAO-covariant response``.** :class:`PaoCovariantResponse` cannot be
  constructed without a :class:`~epc_basis_response.BasisResponseArtifact`, and
  every contraction in this module goes through it. The overlap-free variant
  lives in :func:`epc_formalism.epc_matrix_elements` behind
  ``control_negative=True`` and has a different ``formalism_id``; it never
  reaches :func:`epc_matrix_element_report`.
* **A synthetic displacement is not a phonon.** The mode side goes through
  :func:`phonon_provider.require_physical_phonon`, so a
  ``synthetic_test_displacement`` cannot produce a ``g_{mn nu}`` (roadmap B7),
  and :func:`artifact_signature.epc_artifact_node` refuses it a second time
  through the DAG ancestry.
* **A Gamma response is not a ``q != 0`` response.** ``q != 0`` raises here,
  pointing at GO-5/D01: the Fourier/atomic-phase convention module does not
  exist yet, and reusing the Gamma machinery under a finite-``q`` label is
  exactly the silent substitution the roadmap forbids (GO-8b).

What the mode does to the derivatives. The raw responses are directional, along
unit-Frobenius patterns ``v_i`` (:class:`fd_perturbation_space.Direction`). A
mode's displacement ``u = sqrt(hbar/2 M omega) e`` is expanded in that set,
``u = sum_i c_i v_i``, with the residual measured and refused above
:data:`SPAN_RESIDUAL_TOL`: a mode outside the span of the computed directions
is a missing calculation, not a rounding error. (F2) is linear in the
responses, so ``g = sum_i c_i g[v_i]`` — one contraction per direction, reused
for every branch and every ``k``.

Signatures. The producers of ``D_H`` and of the basis response must declare the
*same* ``geometry_signature`` and ``basis_signature`` strings, and the response
must name the backend that produced ``D_H``; the phonon must carry the same
geometry signature again. Mismatches raise. The three backends are recorded
separately and ``epc_backend_class`` is derived from them, never accepted from
a caller, so a SIESTA overlap response under a Graph2Mat ``D_H`` signs itself
``hybrid``.

Units: ``D_H`` in eV/Ang, the response in 1/Ang, ``g`` in eV/Ang per unit
displacement and in eV once the mode's zero-point amplitude is applied.
Nothing figure-shaped enters any signature: broadening, meshes and plots live
in ``integrated_observable``, so re-plotting cannot invalidate a contraction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from artifact_signature import (  # noqa: E402
    EPC_BACKEND_CLASS_HYBRID,
    PAO_PROJECTED_MODE_COUPLING,
    epc_artifact_node,
)
from epc_basis_response import BasisResponseArtifact  # noqa: E402
from epc_claim_policy import claim_policy  # noqa: E402
from epc_fourier import (  # noqa: E402
    CONVENTION_ID,
    GAMMA,
    bloch_sum,
    convention_fields,
    require_same_conventions,
)
from epc_formalism import (  # noqa: E402
    DEGRADED_FORMALISM_IDS,
    FORMALISM_ID,
    UNITS,
    EpcBlock,
    epc_matrix_elements,
    gauge_invariant_block_metrics,
    pao_covariant_response,
)
from epc_subspaces import (  # noqa: E402
    block_metrics,
    cluster_groups,
    eigenvalue_resolutions_ev,
    near_degenerate_clusters,
)
from fd_perturbation_space import Direction  # noqa: E402
from phonon_provider import require_physical_phonon  # noqa: E402
from run_deeph_sparse_spectrum import eigenspace_identity_tolerance  # noqa: E402

SCHEMA = "epc_matrix_element_v1"
TICKET = "C19 / E-F_001-S24"

D_H_UNITS = UNITS["D_H"]  # eV/Ang
G_UNITS_PER_DISPLACEMENT = UNITS["g"]  # eV/Ang, per unit-Frobenius pattern
G_UNITS = "eV"  # after the mode's zero-point amplitude

# A mode that is not in the span of the directions whose response was computed
# is a missing calculation. The tolerance is a linear-algebra floor, not a
# physical one: the expansion is exact when the direction set spans the mode.
SPAN_RESIDUAL_TOL = 1e-9

# Matrix products between S and C†SC on the dense Loewdin path of
# :func:`dense_eigenspace`: S^{-1/2} (two), S^{-1/2} H S^{-1/2} (two more).
DENSE_LOEWDIN_PRODUCTS = 4

# What may be claimed, given what the basis response still omits (memo A2 /
# C14C): the intra-atomic term is invisible to dS/dR, so a response assembled
# without it bounds g rather than measuring it (roadmap claim ladder).
RESULT_QUANTITATIVE = "quantitative_g"
RESULT_BOUND = "bound__intra_atomic_basis_response_unresolved"
RESULT_DIAGNOSTIC = "diagnostic__degraded_formalism"


class EpcContractionError(RuntimeError):
    """A contraction was attempted that the formalism or the DAG forbids."""


def _finite(name: str, array: Any) -> np.ndarray:
    values = np.asarray(array)
    if not np.all(np.isfinite(values)):
        raise EpcContractionError(f"{name} has non-finite entries")
    return values


# --------------------------------------------------------------------------- #
# The raw electronic response
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RawDerivative:
    """``D_H[v]`` (and optionally ``D_S[v]``) as real-space blocks, signed.

    One directional derivative of the matrices along one unit-Frobenius
    ``direction``, in the repository's cell gauge: ``blocks[name]`` has shape
    ``(n_s, no_u, no_u)`` and ``isc_off[l]`` is the integer lattice vector of
    image ``l``, exactly as :class:`epc_basis_response.BasisResponseArtifact`
    stores its own matrices.

    ``D_S`` is optional and is *not* a basis response: it is the sum
    ``S_L + S_R``, which fixes only the hermitian part of the connection. It is
    carried because the generalized Hellmann-Feynman diagonal is the one
    identity that holds without the split (:func:`hellmann_feynman_diagonal`).
    """

    blocks: dict[str, np.ndarray]
    isc_off: np.ndarray
    backend: str
    method: str  # jvp | frozen_central_difference | siesta_fc_dHS
    direction: Direction
    geometry_signature: str
    basis_signature: str
    topology_sha256: str
    delta_or_jvp: Any
    q: tuple[float, float, float] = GAMMA
    q_aware_source: bool = False
    units: str = D_H_UNITS

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise EpcContractionError(
                "direction must be a fd_perturbation_space.Direction; an unnamed, "
                "unnormalised pattern cannot be matched against a basis response"
            )
        for name in ("backend", "method", "geometry_signature", "basis_signature"):
            if not str(getattr(self, name) or "").strip():
                raise EpcContractionError(f"{name} must be declared; an unsigned D_H is unusable")
        if "D_H" not in self.blocks:
            raise EpcContractionError(f"blocks must contain 'D_H', got {sorted(self.blocks)}")
        if set(self.blocks) - {"D_H", "D_S"}:
            raise EpcContractionError(
                f"a raw derivative carries D_H and optionally D_S, got {sorted(self.blocks)}. "
                "S_L/S_R belong to the basis-response artifact, which is signed separately"
            )
        isc = np.asarray(self.isc_off, dtype=np.int64)
        if isc.ndim != 2 or isc.shape[1] != 3:
            raise EpcContractionError(f"isc_off must be (n_s, 3) integer offsets, got {isc.shape}")
        blocks = {}
        for name, values in self.blocks.items():
            array = _finite(name, values)
            if array.ndim == 2:
                array = array[None, :, :]
            if array.ndim != 3 or array.shape[1] != array.shape[2]:
                raise EpcContractionError(
                    f"{name} must be (n_s, no_u, no_u) blocks or one square matrix, "
                    f"got shape {array.shape}"
                )
            if array.shape[0] != isc.shape[0]:
                raise EpcContractionError(
                    f"{name} carries {array.shape[0]} images but isc_off has {isc.shape[0]}"
                )
            target = np.complex128 if np.issubdtype(array.dtype, np.complexfloating) else np.float64
            blocks[name] = array.astype(target, copy=False)
        if len({values.shape for values in blocks.values()}) != 1:
            raise EpcContractionError("D_H and D_S disagree in shape")
        q = tuple(float(value) for value in self.q)
        if len(q) != 3:
            raise EpcContractionError(f"q must be a fractional 3-vector, got {self.q!r}")
        if any(q) and not self.q_aware_source:
            raise EpcContractionError(
                f"q = {q} but the source is not q-aware: a cell-periodic (q = 0) JVP cannot be "
                "relabelled at finite q (GO-8b)"
            )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "isc_off", isc)
        object.__setattr__(self, "q", q)

    @property
    def no_u(self) -> int:
        return int(self.blocks["D_H"].shape[1])

    @property
    def dtype(self) -> str:
        return str(self.blocks["D_H"].dtype)

    @property
    def has_D_S(self) -> bool:
        return "D_S" in self.blocks

    def at_k(self, k: Any = GAMMA) -> dict[str, np.ndarray]:
        """Bloch sum every block at fractional ``k`` (:data:`BLOCH_CONVENTION`)."""
        return {name: bloch_sum(values, self.isc_off, k) for name, values in self.blocks.items()}

    def conventions(self) -> dict[str, Any]:
        """The S29 q-space record (:mod:`epc_fourier`), same one the response uses."""
        return {
            **convention_fields(self.q, q_aware_source=self.q_aware_source),
            "geometry_signature": self.geometry_signature,
            "basis_signature": self.basis_signature,
        }

    def artifact_fields(self) -> dict[str, Any]:
        """Exactly the fields the ``raw_derivative`` DAG kind declares."""
        return {
            "derivative_backend": self.backend,
            "derivative_method": self.method,
            "perturbation_definition": self.direction.to_metadata(),
            "q": list(self.q),
            "delta_or_jvp": self.delta_or_jvp,
            "topology_sha256": self.topology_sha256,
            "convention": self.conventions(),
            "dtype": self.dtype,
            "units": self.units,
        }

    def signature_node(
        self,
        *,
        hamiltonian: dict[str, Any],
        overlap: dict[str, Any] | None = None,
        mode: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return epc_artifact_node(
            "raw_derivative",
            self.artifact_fields(),
            {"hamiltonian": hamiltonian, "overlap": overlap, "mode": mode},
        )


# --------------------------------------------------------------------------- #
# Raw response + basis response = the PAO-covariant response
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PaoCovariantResponse:
    """``PAO-covariant response`` of one direction: the only object that may be contracted.

    This is the covariant response of the finite, atom-centered PAO subspace
    -- not proven equal to the full KS Hamiltonian derivative matrix element.
    The intra-atomic basis-response term and the ``(1-P)`` (``Delta_out``)
    residual are not quantified (memo section 6 / C14C), so "PAO-covariant"
    is the honest ceiling of what this object claims, not "physical".

    Construction *is* the gate. There is no default response, no ``None``
    fallback and no "ignore overlap" flag: a caller holding only ``D_H`` cannot
    build this class, which is what makes "``D_H`` treated as ``PAO-covariant response``"
    unreachable rather than discouraged.
    """

    raw: RawDerivative
    response: BasisResponseArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.response, BasisResponseArtifact):
            raise EpcContractionError(
                "a PAO-covariant response needs a BasisResponseArtifact: without it "
                "C^dag D_H C is not g, and is not even invariant under H -> H + cS"
            )
        context = self.response.context
        mismatches = {
            name: (mine, theirs)
            for name, mine, theirs in (
                ("geometry_signature", self.raw.geometry_signature, context.geometry_signature),
                ("basis_signature", self.raw.basis_signature, context.basis_signature),
                (
                    "electronic_derivative_backend",
                    self.raw.backend,
                    context.electronic_derivative_backend,
                ),
                ("q", list(self.raw.q), list(context.q)),
                ("perturbation_definition", self.raw.direction.to_metadata(), context.definition),
            )
            if mine != theirs
        }
        if mismatches:
            raise EpcContractionError(
                f"the raw derivative and the basis response are not the same calculation: "
                f"{mismatches}. Both producers must declare the same geometry/basis signature "
                f"scheme, the same displacement pattern and the same q"
            )
        # ... and the same phases: the basis response is q-aware under exactly the
        # convention D_H is (S29). A q-aware D_H with a Gamma-phased overlap
        # response is the failure GO-5 names explicitly.
        require_same_conventions(
            self.raw.conventions(),
            self.response.conventions(),
            what="the raw derivative and the basis response",
        )
        if self.raw.no_u != self.response.no_u:
            raise EpcContractionError(
                f"D_H has no_u = {self.raw.no_u}, the response has {self.response.no_u}"
            )

    # -- provenance ----------------------------------------------------------
    @property
    def formalism_id(self) -> str:
        return self.response.context.formalism_id

    @property
    def is_degraded(self) -> bool:
        return self.formalism_id in DEGRADED_FORMALISM_IDS

    @property
    def electronic_derivative_backend(self) -> str:
        return self.raw.backend

    @property
    def basis_response_backend(self) -> str:
        return self.response.backend

    @property
    def intra_atomic_included(self) -> bool:
        return bool(self.response.intra_atomic_included)

    @property
    def direction(self) -> Direction:
        return self.raw.direction

    def artifact_fields(self) -> dict[str, Any]:
        """Exactly the fields the ``pao_covariant_response`` DAG kind declares."""
        return {
            "formalism_id": self.formalism_id,
            "basis_response_convention": self.response.conventions(),
            "electronic_derivative_backend": self.electronic_derivative_backend,
            "basis_response_backend": self.basis_response_backend,
            "units": D_H_UNITS,
        }

    def signature_node(
        self, *, raw_derivative: dict[str, Any], basis_response: dict[str, Any]
    ) -> dict[str, Any]:
        return epc_artifact_node(
            "pao_covariant_response",
            self.artifact_fields(),
            {"raw_derivative": raw_derivative, "basis_response": basis_response},
        )

    # -- the physics ---------------------------------------------------------
    def matrices_at_k(self, k: Any = GAMMA) -> tuple[np.ndarray, Any]:
        """``(D_H(k), BasisResponse(k))``, the two halves of (F1)/(F2)."""
        return self.raw.at_k(k)["D_H"], self.response.at_k(k)

    def pao_covariant_response(self, H_at_k: Any, S_at_k: Any, k: Any = GAMMA) -> np.ndarray:
        """Memo (F1) as an explicit matrix. Small-system/reference path only.

        It inverts ``S``; :meth:`block` does not, because ``H C = S C eps``
        cancels both inverses. Use this to *inspect* ``PAO-covariant response``, not to
        produce a production ``g`` at MATBG size.
        """
        d_h, response = self.matrices_at_k(k)
        return pao_covariant_response(d_h, H_at_k, S_at_k, response)

    def block(self, row: "Eigenspace", col: "Eigenspace") -> EpcBlock:
        """Memo (F2) between two S-orthonormal windows, per unit displacement."""
        if not np.allclose(row.k, col.k, atol=1e-12):
            raise EpcContractionError(
                f"row k = {row.k} and column k = {col.k} differ, which is a q != 0 "
                f"contraction; see the q gate in this module (GO-5/D01)"
            )
        d_h, response = self.matrices_at_k(col.k)
        if d_h.shape[0] != row.no_u or d_h.shape[0] != col.no_u:
            raise EpcContractionError(
                f"D_H is {d_h.shape} but the windows carry {row.no_u} and {col.no_u} orbitals"
            )
        return epc_matrix_elements(
            d_h,
            response,
            C_row=row.C,
            C_col=col.C,
            eps_row=row.eps,
            eps_col=col.eps,
        )


# --------------------------------------------------------------------------- #
# The electrons
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Eigenspace:
    """One S-orthonormal window: ``C``, ``eps`` and the error they carry.

    ``C†SC = I`` only to ``identity_tolerance`` (C15's own budget), and a window
    that misses it is rejected here rather than producing a ``g`` whose gauge
    error nobody measured.
    """

    label: str
    k: tuple[float, float, float]
    C: np.ndarray
    eps: np.ndarray
    identity_error: float
    identity_tolerance: float
    resolutions_ev: np.ndarray
    cluster_labels: list[int]
    node: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        c = np.asarray(_finite("C", self.C))
        eps = np.asarray(_finite("eps", self.eps), dtype=np.float64).reshape(-1)
        if c.ndim != 2:
            raise EpcContractionError(f"C must be (no_u, n_states), got {c.shape}")
        if eps.size != c.shape[1]:
            raise EpcContractionError(f"eps carries {eps.size} values for {c.shape[1]} states")
        if len(self.cluster_labels) != eps.size:
            raise EpcContractionError(
                f"{len(self.cluster_labels)} cluster labels for {eps.size} states"
            )
        if self.identity_error > self.identity_tolerance:
            raise EpcContractionError(
                f"window {self.label!r} is not S-orthonormal: |C†SC - I| = "
                f"{self.identity_error:.3e} exceeds its own budget "
                f"{self.identity_tolerance:.3e}. g contracted between non-orthonormal "
                f"states is not a coupling"
            )
        object.__setattr__(self, "C", c)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "k", tuple(float(value) for value in np.reshape(self.k, -1)))
        object.__setattr__(
            self, "resolutions_ev", np.asarray(self.resolutions_ev, dtype=np.float64).reshape(-1)
        )

    @property
    def no_u(self) -> int:
        return int(self.C.shape[0])

    @property
    def state_count(self) -> int:
        return int(self.eps.size)

    @classmethod
    def from_persisted(
        cls, payload: dict[str, Any], *, label: str = "", separation_factor: float = 1.0
    ) -> "Eigenspace":
        """Wrap one C15 window (``run_deeph_sparse_spectrum.load_persisted_eigenspace``)."""
        energies = np.asarray(payload["energies_eV"], dtype=np.float64)
        residuals = np.asarray(payload["generalized_relative_residual"], dtype=np.float64)
        condition = float(payload["window_metric_condition_number"])
        orbital_count = int(payload["norbits"])
        deviation = np.asarray(payload["overlap_minus_identity"])
        resolutions = eigenvalue_resolutions_ev(energies, residuals, condition, orbital_count)
        return cls(
            label=label or f"k{int(payload['k_index']):03d}",
            k=tuple(float(value) for value in payload["k_fractional"]),
            C=payload["coefficients"],
            eps=energies,
            identity_error=float(np.abs(deviation).max(initial=0.0)),
            identity_tolerance=eigenspace_identity_tolerance(
                float(residuals.max(initial=0.0)), condition, orbital_count
            ),
            resolutions_ev=resolutions,
            cluster_labels=near_degenerate_clusters(
                energies, resolutions, separation_factor=separation_factor
            ),
            node=payload.get("signature"),
            diagnostics={
                "k_index": int(payload["k_index"]),
                "window_metric_condition_number": condition,
                "maximum_generalized_relative_residual": float(residuals.max(initial=0.0)),
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "k": list(self.k),
            "state_count": self.state_count,
            "energies_eV": self.eps.tolist(),
            "resolutions_eV": self.resolutions_ev.tolist(),
            "cluster_labels": list(self.cluster_labels),
            "identity_maximum_absolute_error": self.identity_error,
            "identity_tolerance": self.identity_tolerance,
            **self.diagnostics,
        }


def dense_eigenspace(
    k: Any,
    H: Any,
    S: Any,
    *,
    label: str,
    window: slice | None = None,
    separation_factor: float = 1.0,
) -> Eigenspace:
    """Solve ``H C = S C eps`` densely and wrap it. Fixtures and small systems.

    Loewdin: ``C = S^{-1/2} V`` with ``V`` the eigenvectors of
    ``S^{-1/2} H S^{-1/2}``, which is S-orthonormal by construction. Production
    windows come from the sparse solver through :meth:`Eigenspace.from_persisted`.

    The identity budget is C15's policy times :data:`DENSE_LOEWDIN_PRODUCTS`:
    this path chains that many ``n``-term matrix products between ``S`` and
    ``C†SC``, and each one contributes its own ``n * eps * cond`` term. Same
    policy, different arithmetic — not a looser tolerance chosen to pass.
    """
    h = np.asarray(_finite("H", H))
    s = np.asarray(_finite("S", S))
    metric_eigenvalues, metric_vectors = np.linalg.eigh(s)
    if metric_eigenvalues.min() <= 0.0:
        raise EpcContractionError(
            f"S is not positive definite (min eigenvalue {metric_eigenvalues.min():.3e})"
        )
    inverse_root = (metric_vectors * metric_eigenvalues**-0.5) @ metric_vectors.conj().T
    eps, vectors = np.linalg.eigh(inverse_root.conj().T @ h @ inverse_root)
    coefficients = inverse_root @ vectors
    if window is not None:
        eps, coefficients = eps[window], coefficients[:, window]
    residual = np.linalg.norm(h @ coefficients - (s @ coefficients) * eps[None, :], axis=0)
    scale = np.linalg.norm(h @ coefficients, axis=0) + np.abs(eps) * np.linalg.norm(
        s @ coefficients, axis=0
    )
    relative_residual = residual / np.where(scale > 0.0, scale, 1.0)
    condition = float(metric_eigenvalues.max() / metric_eigenvalues.min())
    orbital_count = int(h.shape[0])
    resolutions = eigenvalue_resolutions_ev(eps, relative_residual, condition, orbital_count)
    identity = coefficients.conj().T @ s @ coefficients
    return Eigenspace(
        label=label,
        k=k,
        C=coefficients,
        eps=eps,
        identity_error=float(np.abs(identity - np.eye(eps.size)).max(initial=0.0)),
        identity_tolerance=DENSE_LOEWDIN_PRODUCTS
        * eigenspace_identity_tolerance(
            float(relative_residual.max(initial=0.0)), condition, orbital_count
        ),
        resolutions_ev=resolutions,
        cluster_labels=near_degenerate_clusters(
            eps, resolutions, separation_factor=separation_factor
        ),
        diagnostics={
            "window_metric_condition_number": condition,
            "maximum_generalized_relative_residual": float(relative_residual.max(initial=0.0)),
            "solver": "dense_loewdin_numpy",
        },
    )


# --------------------------------------------------------------------------- #
# Contraction
# --------------------------------------------------------------------------- #


def _homogeneous(perturbations: Sequence[PaoCovariantResponse]) -> PaoCovariantResponse:
    """All directions must come from the same formalism, backends and signatures."""
    if not perturbations:
        raise EpcContractionError(
            "no PAO-covariant response was given; there is nothing to contract"
        )
    first = perturbations[0]
    keys = [
        (
            p.formalism_id,
            p.response.representation,
            p.electronic_derivative_backend,
            p.basis_response_backend,
            p.raw.geometry_signature,
            p.raw.basis_signature,
            p.intra_atomic_included,
            tuple(p.raw.q),
        )
        for p in perturbations
    ]
    if len(set(keys)) != 1:
        raise EpcContractionError(
            "the directions of one mode must share formalism, representation, backends, "
            f"signatures, intra-atomic status and q; got {sorted(set(keys))}"
        )
    return first


def direction_coefficients(
    perturbations: Sequence[PaoCovariantResponse],
    pattern: Any,
    *,
    residual_tol: float = SPAN_RESIDUAL_TOL,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expand a displacement ``[na, 3]`` in the directions that were computed.

    ``u = sum_i c_i v_i`` by least squares, with the residual measured. Above
    ``residual_tol`` the mode leaves the span of the computed responses and the
    expansion raises: the missing piece is a calculation nobody ran, and
    projecting the mode onto what happens to be available would silently
    replace it with a different perturbation.
    """
    _homogeneous(perturbations)
    displacement = np.asarray(_finite("pattern", pattern))
    if displacement.ndim != 2 or displacement.shape[1] != 3:
        raise EpcContractionError(f"the displacement must be [na, 3], got {displacement.shape}")
    basis = np.stack([p.direction.vectors for p in perturbations])
    if basis.shape[1:] != displacement.shape:
        raise EpcContractionError(
            f"the directions act on {basis.shape[1]} atoms, the displacement on "
            f"{displacement.shape[0]}"
        )
    matrix = basis.reshape(len(perturbations), -1).T  # (3 na, n_directions)
    target = displacement.reshape(-1)
    coefficients, *_ = np.linalg.lstsq(matrix.astype(target.dtype), target, rcond=None)
    residual = float(np.linalg.norm(matrix @ coefficients - target))
    norm = float(np.linalg.norm(target))
    relative = residual / norm if norm > 0.0 else residual
    if relative > residual_tol:
        raise EpcContractionError(
            f"the displacement is not in the span of the {len(perturbations)} directions whose "
            f"response was computed (relative residual {relative:.3e} > {residual_tol:.1e}). "
            f"Compute D_H along the missing direction instead of projecting the mode onto the "
            f"responses that happen to exist"
        )
    return coefficients, {
        "direction_names": [p.direction.name for p in perturbations],
        "direction_hashes": [p.direction.direction_hash for p in perturbations],
        "coefficients_real": np.real(coefficients).tolist(),
        "coefficients_imag": np.imag(coefficients).tolist(),
        "span_relative_residual": relative,
        "span_residual_tolerance": float(residual_tol),
        "pattern_frobenius": norm,
    }


def contract(
    perturbations: Sequence[PaoCovariantResponse],
    coefficients: Any,
    row: Eigenspace,
    col: Eigenspace,
    *,
    units: str = G_UNITS_PER_DISPLACEMENT,
) -> EpcBlock:
    """``g = sum_i c_i g[v_i]``: (F2) per direction, combined linearly.

    (F2) is linear in ``D_H``, ``S_L`` and ``S_R``, so one contraction per
    direction serves every branch and every amplitude — the expensive object is
    the derivative, not the coupling.
    """
    first = _homogeneous(perturbations)
    weights = np.asarray(coefficients).reshape(-1)
    if weights.size != len(perturbations):
        raise EpcContractionError(
            f"{weights.size} coefficients for {len(perturbations)} directions"
        )
    blocks = [p.block(row, col) for p in perturbations]
    values = sum(weight * block.values for weight, block in zip(weights, blocks))
    return EpcBlock(
        values=np.asarray(values),
        formalism_id=blocks[0].formalism_id,
        representation=first.response.representation,
        intra_atomic_included=first.intra_atomic_included,
        basis_response_backend=first.basis_response_backend,
        units=units,
    )


def hellmann_feynman_diagonal(
    perturbation: PaoCovariantResponse, eigenspace: Eigenspace
) -> dict[str, Any]:
    """``d(eps_n) = C_n† (D_H - eps_n D_S) C_n`` against the diagonal of (F2).

    The two are the same number by algebra — ``S_L + S_R = D_S`` and the two
    energy weights of (F2) collapse on the diagonal — which is why this check
    needs no split and survives the unresolved intra-atomic term. What it
    catches is everything that is *not* algebra: a response Bloch-summed at the
    wrong ``k``, a transposed ``S_L``/``S_R``, an eV/Ry mix-up.

    ``D_S`` is taken from the raw derivative when it carries one (an
    independent measurement) and from the response's own sum otherwise; the
    source is reported, because only the first version has teeth.
    """
    block = perturbation.block(eigenspace, eigenspace)
    matrices = perturbation.raw.at_k(eigenspace.k)
    if perturbation.raw.has_D_S:
        d_s, source = matrices["D_S"], "raw_derivative"
    else:
        d_s, source = perturbation.response.at_k(eigenspace.k).total, "basis_response_sum"
    coefficients, energies = eigenspace.C, eigenspace.eps
    projected_h = np.einsum("in,ij,jn->n", coefficients.conj(), matrices["D_H"], coefficients)
    projected_s = np.einsum("in,ij,jn->n", coefficients.conj(), d_s, coefficients)
    reference = projected_h - energies * projected_s
    diagonal = np.diag(np.asarray(block.values))
    residual = float(np.max(np.abs(diagonal - reference), initial=0.0))
    scale = float(np.max(np.abs(reference), initial=0.0))
    # Pure roundoff: no_u inner-product terms, amplified by the window's own
    # gauge error, on the largest energy in play.
    tolerance = float(
        eigenspace.no_u
        * np.finfo(np.float64).eps
        * max(scale, 1.0)
        * max(1.0, float(np.max(np.abs(energies), initial=0.0)))
        + eigenspace.identity_tolerance * max(scale, 1.0)
    )
    return {
        "identity": "diag(F2) == C_n^dag (D_H - eps_n D_S) C_n",
        "d_s_source": source,
        "values_eV_per_Ang": np.real(reference).tolist(),
        "maximum_absolute_imaginary_part": float(
            np.max(np.abs(np.imag(reference)), initial=0.0)
        ),
        "residual": residual,
        "reference_frobenius": scale,
        "tolerance": tolerance,
        "passes": bool(residual <= tolerance),
    }


def finite_difference_eigenvalue_derivatives(
    H: Any, S: Any, D_H: Any, D_S: Any, *, delta: float = 1e-4, window: slice | None = None
) -> np.ndarray:
    """``d(eps)/dt`` of the pencil ``(H + t D_H, S + t D_S)`` by central difference.

    The physical half of the generalized Hellmann-Feynman check: the diagonal
    of (F2) has to reproduce *this*, and on a dense fixture there is no reason
    it should unless every convention lines up.
    """
    h, s, d_h, d_s = (np.asarray(value) for value in (H, S, D_H, D_S))
    plus = dense_eigenspace(GAMMA, h + delta * d_h, s + delta * d_s, label="fd+", window=window)
    minus = dense_eigenspace(GAMMA, h - delta * d_h, s - delta * d_s, label="fd-", window=window)
    return (plus.eps - minus.eps) / (2.0 * delta)


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #


def _block_payload(
    block: EpcBlock, eigenspace_row: Eigenspace, eigenspace_col: Eigenspace
) -> dict[str, Any]:
    values = np.asarray(block.values)
    return {
        "units": block.units,
        # The direct matrix elements. Band-indexed, hence gauge: readable only
        # together with the cluster labels below.
        "values_real": np.real(values).tolist(),
        "values_imag": np.imag(values).tolist(),
        "gauge_invariant": gauge_invariant_block_metrics(block),
        "cluster_blocks": block_metrics(
            values, eigenspace_row.cluster_labels, eigenspace_col.cluster_labels
        ),
        "cluster_groups_row": cluster_groups(eigenspace_row.cluster_labels),
        "cluster_groups_column": cluster_groups(eigenspace_col.cluster_labels),
    }


def epc_matrix_element_report(
    perturbations: Sequence[PaoCovariantResponse],
    mode_set: Any,
    branch: int,
    eigenspace_k: Eigenspace,
    eigenspace_k_plus_q: Eigenspace | None = None,
    *,
    label: str = "",
    perturbation_nodes: Sequence[dict[str, Any]] | None = None,
    phonon_node: dict[str, Any] | None = None,
    span_residual_tol: float = SPAN_RESIDUAL_TOL,
) -> dict[str, Any]:
    """``g_{mn nu}`` of one branch, with the provenance that makes it citable.

    ``mode_set`` must be a :class:`phonon_provider.PhononModeSet` in the
    canonical conventions; a ``SyntheticTestDisplacement`` is refused here and
    again by the DAG. ``perturbation_nodes`` are the signed
    ``pao_covariant_response`` nodes of the directions, in the same order; give
    them together with ``phonon_node`` to get a signed
    ``pao_projected_mode_coupling`` node back.
    """
    first = _homogeneous(perturbations)
    modes = require_physical_phonon(mode_set)
    if not modes.conventions.is_canonical:
        raise EpcContractionError(
            f"the mode set is in {modes.conventions.to_dict()}; convert it with "
            "phonon_provider.to_canonical_conventions() before contracting, so the phase "
            "conversion is represented instead of applied ad hoc"
        )
    q = tuple(float(value) for value in modes.q_fractional)
    if any(q):
        raise EpcContractionError(
            f"q = {q}: the conventions are fixed (epc_fourier, {CONVENTION_ID}) but no q-aware "
            "D_H/basis-response source is certified yet — GO-5 (graphene K) is what licenses "
            "one, and the Gamma response may not stand in for it (GO-8b)"
        )
    if tuple(first.raw.q) != q:
        raise EpcContractionError(
            f"the responses were computed at q = {first.raw.q} but the mode is at q = {q}"
        )
    if not str(modes.geometry_signature or "").strip():
        raise EpcContractionError(
            "the phonon carries no geometry_signature; an unsigned mode cannot be shown to "
            "belong to the geometry whose D_H was differentiated"
        )
    if modes.geometry_signature != first.raw.geometry_signature:
        raise EpcContractionError(
            f"the phonon is signed {modes.geometry_signature!r} and the derivatives "
            f"{first.raw.geometry_signature!r}: different geometries, or two producers using "
            f"different geometry-signature schemes. Either way the contraction is not defined"
        )

    mode = modes.mode(int(branch))
    if mode.eigenvector.shape[0] != first.direction.atom_count:
        raise EpcContractionError(
            f"the mode moves {mode.eigenvector.shape[0]} atoms, the directions "
            f"{first.direction.atom_count}"
        )
    row = eigenspace_k_plus_q or eigenspace_k
    if row.no_u != first.raw.no_u or eigenspace_k.no_u != first.raw.no_u:
        raise EpcContractionError(
            f"the windows carry {eigenspace_k.no_u}/{row.no_u} orbitals, D_H has "
            f"{first.raw.no_u}"
        )

    # Per unit displacement first: it is the object the tolerances of the
    # pre-registration are defined on, and it does not depend on the amplitude.
    pattern = mode.displacement_pattern()  # Ang, zero-point amplitude included
    amplitude_coefficients, expansion = direction_coefficients(
        perturbations, pattern, residual_tol=span_residual_tol
    )
    g_ev = contract(perturbations, amplitude_coefficients, row, eigenspace_k, units=G_UNITS)
    pattern_norm = expansion["pattern_frobenius"]
    g_per_ang = contract(
        perturbations,
        amplitude_coefficients / pattern_norm,
        row,
        eigenspace_k,
        units=G_UNITS_PER_DISPLACEMENT,
    )

    backends = {
        "electronic_derivative_backend": first.electronic_derivative_backend,
        "basis_response_backend": first.basis_response_backend,
        "phonon_backend": modes.provider,
    }
    epc_backend_class = (
        first.electronic_derivative_backend
        if len(set(backends.values())) == 1
        else EPC_BACKEND_CLASS_HYBRID
    )
    if first.is_degraded:
        result_class = RESULT_DIAGNOSTIC
    elif not first.intra_atomic_included:
        result_class = RESULT_BOUND
    else:
        result_class = RESULT_QUANTITATIVE

    phonon_fields = modes.physical_phonon_fields()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": label or f"{first.direction.name}__branch{int(branch)}",
        "formalism_id": first.formalism_id,
        "basis_response_representation": first.response.representation,
        "observable": "g_PAO from Delta_PAO_cov (finite-PAO covariant response)",
        "claim_policy": claim_policy({}),
        "intra_atomic_included": first.intra_atomic_included,
        "result_class": result_class,
        "degraded_formalism": DEGRADED_FORMALISM_IDS.get(first.formalism_id),
        **backends,
        "epc_backend_class": epc_backend_class,
        "mode": {
            "artifact_kind": modes.artifact_kind,
            "provider": modes.provider,
            "provider_version": modes.provider_version,
            "branch": int(branch),
            "q": list(q),
            "frequency_eV": float(mode.frequency_ev),
            "zero_point_amplitude_Ang": mode.zero_point_amplitude_ang.tolist(),
            "displacement_frobenius_Ang": pattern_norm,
            "normalization": phonon_fields["normalization"],
            "fourier_convention": phonon_fields["fourier_convention"],
            "asr_policy": phonon_fields["asr_policy"],
            "fc_range": phonon_fields["fc_range"],
        },
        "expansion": expansion,
        "electronic_states": {
            "k": eigenspace_k.summary(),
            "k_plus_q": row.summary(),
        },
        "g": _block_payload(g_ev, row, eigenspace_k),
        "g_per_unit_displacement": _block_payload(g_per_ang, row, eigenspace_k),
    }
    if row is eigenspace_k or np.allclose(row.k, eigenspace_k.k, atol=1e-12):
        report["hellmann_feynman"] = [
            hellmann_feynman_diagonal(p, eigenspace_k) for p in perturbations
        ]

    if perturbation_nodes is not None and phonon_node is not None:
        if len(perturbation_nodes) != len(perturbations):
            raise EpcContractionError(
                f"{len(perturbation_nodes)} perturbation nodes for {len(perturbations)} directions"
            )
        report["signatures"] = [
            _signature_node(
                perturbation_node,
                phonon_node,
                eigenspace_k,
                row,
                phonon_fields=phonon_fields,
                backends=backends,
                epc_backend_class=epc_backend_class,
            )
            for perturbation_node in perturbation_nodes
        ]
    return report


def _signature_node(
    perturbation_node: dict[str, Any],
    phonon_node: dict[str, Any],
    eigenspace_k: Eigenspace,
    eigenspace_k_plus_q: Eigenspace,
    *,
    phonon_fields: dict[str, Any],
    backends: dict[str, str],
    epc_backend_class: str,
) -> dict[str, Any]:
    """The signed ``pao_projected_mode_coupling`` node of one direction's contribution.

    Nothing presentational is in ``fields``: broadening, meshes and figures are
    ``integrated_observable``'s business, so re-plotting a result cannot move
    this hash — and ``epc_artifact_node`` rejects any extra field that tried.
    """
    for name, space in (("k", eigenspace_k), ("k_plus_q", eigenspace_k_plus_q)):
        if space.node is None:
            raise EpcContractionError(
                f"the {name} window carries no electronic_eigenspace signature node; "
                "persist it with the solver (C15) before signing a g"
            )
    return epc_artifact_node(
        PAO_PROJECTED_MODE_COUPLING,
        {
            "k": list(eigenspace_k.k),
            "k_plus_q": list(eigenspace_k_plus_q.k),
            "fourier_convention": phonon_fields["fourier_convention"],
            "phonon_normalization": phonon_fields["normalization"],
            "phonon_backend": backends["phonon_backend"],
            "epc_backend_class": epc_backend_class,
            "units": G_UNITS,
        },
        {
            "perturbation": perturbation_node,
            "phonon": phonon_node,
            "eigenspace_k": eigenspace_k.node,
            "eigenspace_k_plus_q": eigenspace_k_plus_q.node,
        },
    )


def contraction_contract() -> dict[str, Any]:
    """The C19 contract as data, for manifests and the UI (no physics here)."""
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "formalism_id": FORMALISM_ID,
        "pao_covariant_response": "built only from a raw D_H and a basis-response artifact of the same "
        "direction, geometry and basis signature (F1); contracted through (F2)",
        "g": "g = sum_i c_i g[v_i] with u = sum_i c_i v_i the mode's displacement pattern",
        "units": {
            "D_H": D_H_UNITS,
            "g_per_unit_displacement": G_UNITS_PER_DISPLACEMENT,
            "g": G_UNITS,
        },
        "refusals": {
            "d_h_as_pao_covariant_response": "PaoCovariantResponse requires a BasisResponseArtifact",
            "synthetic_displacement": "require_physical_phonon plus the DAG ancestry rule",
            "finite_q": "GO-5/D01 must fix the Fourier convention first",
            "mode_outside_span": f"relative residual > {SPAN_RESIDUAL_TOL}",
            "non_orthonormal_window": "|C†SC - I| above C15's own budget",
        },
        "backends_recorded": (
            "electronic_derivative_backend",
            "basis_response_backend",
            "phonon_backend",
            "epc_backend_class",
        ),
        "result_classes": (RESULT_QUANTITATIVE, RESULT_BOUND, RESULT_DIAGNOSTIC),
    }
