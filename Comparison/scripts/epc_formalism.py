#!/usr/bin/env python3
"""C04 / E-F_001-S6: the moving-PAO EPC algebraic contract, as code.

This module is the executable half of ``docs/epc_formalismo_pao_movil.md``
(C03b / GO-1). It encodes one thing: how the *raw* matrix responses a backend
produces become the finite-PAO covariant perturbation contracted between
generalized eigenvectors, and refuses every shortcut the memo rules out.

    formalism_id = "moving_pao_projected_dual_basis_v1"

    (F1)  PAO-covariant response[u] = D_H[u] - S_L[u] S^-1 H - H S^-1 S_R[u]
    (F2)  g_mn          = (Cm^dag D_H Cn) - eps_n (Cm^dag S_L Cn)
                                          - eps_m (Cm^dag S_R Cn)

with ``S_L[u]_{mu,nu} = <d_u phi_mu | phi_nu>`` and
``S_R[u]_{mu,nu} = <phi_mu | d_u phi_nu>``. (F2) is (F1) contracted using
``H C = S C eps``, so the production path never inverts or factorises ``S``.

What this module makes impossible, by construction:

* labelling ``D_H`` as ``g``. :func:`epc_matrix_elements` cannot be called
  without a :class:`BasisResponse`, and the overlap-free variant only exists as
  a control negative that must be asked for by name;
* rebuilding ``S_L``/``S_R`` from ``D_S``. A ``D_S``-only response is a
  *different* representation with no ``S_left``/``S_right`` attributes at all;
  it can only produce the degraded symmetric form, tagged as such
  (``D_S`` fixes the hermitian part of ``S Gamma`` and nothing else, memo 4.2);
* omitting the intra-atomic term in silence: ``intra_atomic_included`` is a
  mandatory flag of the response, and it travels into the returned block.

Units are the repository canon: ``H``/``eps`` in eV, ``D_H`` in eV/Ang,
``S_L``/``S_R``/``D_S`` in 1/Ang, so ``g`` is eV/Ang per unit displacement and
becomes eV after multiplying by :func:`zero_point_amplitude_ang`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FORMALISM_ID = "moving_pao_projected_dual_basis_v1"

# Degraded variants: diagnostics only, never production truth (memo section 5).
FORMALISM_ID_SYMMETRIC = "moving_pao_symmetric_overlap_v1"
FORMALISM_ID_IGNORE_OVERLAP = "moving_pao_ignore_overlap_v1"

DEGRADED_FORMALISM_IDS = {
    FORMALISM_ID_SYMMETRIC: "A = (S_L - S_R)/2 assumed zero; exact only on the "
    "diagonal, inside degenerate blocks and in the adiabatic limit",
    FORMALISM_ID_IGNORE_OVERLAP: "basis response dropped entirely; not even "
    "invariant under H -> H + cS. Control negative only",
}

# Values of ``basis_response.representation`` in shared/artifact_signature.py.
REPRESENTATION_S_L_S_R = "S_L_S_R"
REPRESENTATION_D_S = "D_S"

UNITS = {"H": "eV", "D_H": "eV/Ang", "basis_response": "1/Ang", "g": "eV/Ang"}

# hbar^2 / (1 amu * 1 eV) in Ang^2 (memo section 8).
ZERO_POINT_CONSTANT_ANG2_AMU_EV = 4.180160e-3


class EpcFormalismError(ValueError):
    """A caller tried to build a PAO-covariant coupling object the formalism forbids."""


@dataclass(frozen=True)
class BasisResponse:
    """The overlap/basis response of one displacement pattern.

    Build it with :func:`basis_response_S_L_S_R` (production) or
    :func:`basis_response_D_S` (degraded). ``S_left``/``S_right`` are ``None``
    exactly when the representation is ``D_S``: there is no supported way to
    split a sum back into two independent matrices.
    """

    representation: str
    intra_atomic_included: bool
    backend: str
    S_left: np.ndarray | None = None
    S_right: np.ndarray | None = None
    D_S: np.ndarray | None = None

    @property
    def total(self) -> np.ndarray:
        """``D_S = S_L + S_R``, the only combination both representations share."""
        if self.representation == REPRESENTATION_S_L_S_R:
            return self.S_left + self.S_right
        return self.D_S

    @property
    def antisymmetric(self) -> np.ndarray:
        """``A = (S_L - S_R)/2``, the part ``D_S`` cannot carry (memo 4.2/4.7)."""
        if self.representation != REPRESENTATION_S_L_S_R:
            raise EpcFormalismError(
                f"A is not recoverable from representation {self.representation!r}: "
                "D_S only fixes the hermitian part of S*Gamma. Provide S_L and S_R "
                "separately (memo section 6)."
            )
        return 0.5 * (self.S_left - self.S_right)


def _as_matrix(name: str, array: Any) -> np.ndarray:
    matrix = np.asarray(array)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise EpcFormalismError(f"{name} must be a square matrix, got shape {matrix.shape}")
    # ponytail: float64/complex128 only. Single precision loses the 1e-10 the
    # algebraic gates are checked at; a caller that wants float32 says so upstream.
    target = np.complex128 if np.issubdtype(matrix.dtype, np.complexfloating) else np.float64
    return matrix.astype(target, copy=False)


def basis_response_S_L_S_R(
    S_left: Any,
    S_right: Any,
    *,
    intra_atomic_included: bool,
    backend: str,
) -> BasisResponse:
    """The representation the formalism requires (C14B output).

    ``intra_atomic_included`` is not a courtesy flag: the same-atom blocks of
    ``A`` are identically invisible in ``dS/dR`` (memo section 6), so a response
    assembled from ``dHSdR.nc`` alone must declare ``False`` and be treated as
    incomplete downstream.
    """
    left = _as_matrix("S_left", S_left)
    right = _as_matrix("S_right", S_right)
    if left.shape != right.shape:
        raise EpcFormalismError(f"S_left {left.shape} and S_right {right.shape} disagree")
    return BasisResponse(
        representation=REPRESENTATION_S_L_S_R,
        intra_atomic_included=bool(intra_atomic_included),
        backend=str(backend),
        S_left=left,
        S_right=right,
    )


def basis_response_D_S(D_S: Any, *, backend: str) -> BasisResponse:
    """A sum-only response. Degraded: it can only feed the symmetric form."""
    return BasisResponse(
        representation=REPRESENTATION_D_S,
        # A same-atom block contributes nothing to D_S, so "included" is
        # meaningless here and False is the honest value.
        intra_atomic_included=False,
        backend=str(backend),
        D_S=_as_matrix("D_S", D_S),
    )


def check_overlap_response_consistency(
    response: BasisResponse,
    D_S_reference: Any,
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Cross-check a split response against an independently measured ``D_S``.

    Checks ``S_L + S_R == D_S`` and the exact identity ``S_L = S_R^dag``
    (memo section 3). Passing is necessary and *not* sufficient: ``A`` is
    invisible to ``D_S`` by construction, which is why this returns diagnostics
    instead of a licence to reconstruct anything.
    """
    if response.representation != REPRESENTATION_S_L_S_R:
        raise EpcFormalismError("nothing to cross-check: response has no S_L/S_R split")
    reference = _as_matrix("D_S_reference", D_S_reference)
    sum_residual = float(np.linalg.norm(response.total - reference))
    hermitian_residual = float(np.linalg.norm(response.S_left - response.S_right.conj().T))
    scale = float(np.linalg.norm(reference)) or 1.0
    return {
        "sum_residual": sum_residual,
        "hermiticity_residual": hermitian_residual,
        "antisymmetric_norm": float(np.linalg.norm(response.antisymmetric)),
        "passes": sum_residual <= atol * scale and hermitian_residual <= atol * scale,
    }


def pao_covariant_response(
    D_H: Any,
    H: Any,
    S: Any,
    response: BasisResponse,
) -> np.ndarray:
    """Memo Eq. (F1). Reference/small-system path: it does invert ``S``.

    Production contracts with :func:`epc_matrix_elements`, which does not.

    This is the covariant response of the finite PAO subspace, not a proven
    stand-in for the full KS Hamiltonian derivative: the intra-atomic
    basis-response term and the ``(1-P)`` (``Delta_out``) residual are not
    quantified here (memo section 6 / C14C).
    """
    if response.representation != REPRESENTATION_S_L_S_R:
        raise EpcFormalismError(
            f"formalism {FORMALISM_ID} needs S_L and S_R separately; "
            f"representation {response.representation!r} cannot build PAO-covariant response"
        )
    d_h = _as_matrix("D_H", D_H)
    h = _as_matrix("H", H)
    s = _as_matrix("S", S)
    left = np.linalg.solve(s.conj().T, response.S_left.conj().T).conj().T  # S_L S^-1
    right = np.linalg.solve(s, response.S_right)  # S^-1 S_R
    return d_h - left @ h - h @ right


@dataclass(frozen=True)
class EpcBlock:
    """A contracted coupling block plus the provenance that makes it readable."""

    values: np.ndarray
    formalism_id: str
    representation: str
    intra_atomic_included: bool
    basis_response_backend: str
    units: str = UNITS["g"]

    @property
    def is_degraded(self) -> bool:
        return self.formalism_id in DEGRADED_FORMALISM_IDS


def epc_matrix_elements(
    D_H: Any,
    response: BasisResponse,
    *,
    C_row: Any,
    C_col: Any,
    eps_row: Any,
    eps_col: Any,
    control_negative: bool = False,
) -> EpcBlock:
    """Memo Eq. (F2): the coupling block between two sets of eigenvectors.

    ``C_row``/``eps_row`` are the final states (``k+q``), ``C_col``/``eps_col``
    the initial ones (``k``); for Gamma they are the same objects. ``S`` is
    never touched, because ``H C = S C eps`` has already cancelled both
    ``S^-1`` of (F1).

    Which formalism comes out is decided by the response, not by the caller:

    * ``S_L_S_R`` -> (F2), ``formalism_id`` = production;
    * ``D_S`` -> (F3) without the ``A`` term, tagged
      ``moving_pao_symmetric_overlap_v1``;
    * ``response=None`` -> ``C^dag D_H C``, which is not a coupling at all and
      requires ``control_negative=True``.
    """
    d_h = _as_matrix("D_H", D_H)
    c_row = np.asarray(C_row)
    c_col = np.asarray(C_col)
    e_row = np.asarray(eps_row, dtype=np.float64)
    e_col = np.asarray(eps_col, dtype=np.float64)

    raw = c_row.conj().T @ d_h @ c_col

    if response is None:
        if not control_negative:
            raise EpcFormalismError(
                "C^dag D_H C is not g: it is not even invariant under H -> H + cS. "
                f"Pass a BasisResponse, or control_negative=True to get the "
                f"{FORMALISM_ID_IGNORE_OVERLAP!r} diagnostic."
            )
        return EpcBlock(
            values=raw,
            formalism_id=FORMALISM_ID_IGNORE_OVERLAP,
            representation="none",
            intra_atomic_included=False,
            basis_response_backend="none",
        )

    if response.representation == REPRESENTATION_S_L_S_R:
        s_l = c_row.conj().T @ response.S_left @ c_col
        s_r = c_row.conj().T @ response.S_right @ c_col
        values = raw - s_l * e_col[None, :] - s_r * e_row[:, None]
        formalism_id = FORMALISM_ID
    else:
        d_s = c_row.conj().T @ response.total @ c_col
        values = raw - 0.5 * (e_row[:, None] + e_col[None, :]) * d_s
        formalism_id = FORMALISM_ID_SYMMETRIC

    return EpcBlock(
        values=values,
        formalism_id=formalism_id,
        representation=response.representation,
        intra_atomic_included=response.intra_atomic_included,
        basis_response_backend=response.backend,
    )


def gauge_invariant_block_metrics(block: Any) -> dict[str, Any]:
    """What may be published about a (near-)degenerate block.

    Under ``C -> C U`` the block goes to ``U^dag g U``, so band-indexed
    elements are gauge. Singular values and the Frobenius norm are not.
    """
    values = np.asarray(block.values if isinstance(block, EpcBlock) else block)
    singular = np.linalg.svd(values, compute_uv=False)
    return {
        "singular_values": singular.tolist(),
        "frobenius_norm": float(np.linalg.norm(values)),
    }


def zero_point_amplitude_ang(mass_amu: float, frequency_ev: float) -> float:
    """``sqrt(hbar / (2 M omega))`` in Ang, the eV/Ang -> eV factor of a mode."""
    if frequency_ev <= 0.0:
        raise EpcFormalismError(f"frequency must be positive, got {frequency_ev}")
    return float(np.sqrt(ZERO_POINT_CONSTANT_ANG2_AMU_EV / (2.0 * mass_amu * frequency_ev)))


def formalism_contract() -> dict[str, Any]:
    """The GO-1 contract as data, for manifests and the UI (no physics here)."""
    return {
        "formalism_id": FORMALISM_ID,
        "memo": "docs/epc_formalismo_pao_movil.md",
        "D_H": "directional derivative of the H matrix in the co-moving PAO basis",
        "D_S": "S_L + S_R; insufficient on its own",
        "basis_response_representation": REPRESENTATION_S_L_S_R,
        "pao_covariant_response": "D_H - S_L S^-1 H - H S^-1 S_R  (F1)",
        "contraction": "g_mn = Cm^dag D_H Cn - eps_n Cm^dag S_L Cn - eps_m Cm^dag S_R Cn  (F2)",
        "gauge": "C -> C U gives g -> U^dag g U; publish singular values / norms",
        "units": dict(UNITS),
        "degraded_variants": dict(DEGRADED_FORMALISM_IDS),
    }
