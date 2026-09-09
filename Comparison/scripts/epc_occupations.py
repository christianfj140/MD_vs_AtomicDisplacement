#!/usr/bin/env python3
"""E-F_001-S7: occupations and chemical potential, as a declared contract.

Roadmap section XII: every electronic artifact must record electron count, spin
degeneracy, occupation function, electronic temperature/smearing, the ``mu``
convention and whether ``mu`` is externally fixed or follows a fixed ``N``.
This module is that record plus the one formula the fixed-``N`` branch needs,

    (S7)   d(mu) = sum_n w_n f'_n d(eps_n) / sum_n w_n f'_n ,

which is ``dN = 0`` differentiated. Nothing else in the EPC pipeline is allowed
to move the chemical potential.

The four artifacts of section XII, and what they are allowed to know:

===============================  ==========================  ==================
artifact                         smearing?                   d(mu) subtracted?
===============================  ==========================  ==================
``pao_projected_mode_coupling``        no — it is raw ``g_mn``     never
``eigenvalue_directional_...``   no — only the energy zero   never
``deformation_potential_rel...`` yes — needs f'              yes, fixed-N only
``fermi_surface_response``       yes — it *is* d(mu)         it is the answer
===============================  ==========================  ==================

That split is what makes re-smearing cheap: it invalidates the bottom two rows
and nothing above them (memo section 9 — a shift ``H -> H + cS`` cancels
identically in ``g``, so only the diagonal quantities see the energy zero at
all). The DAG half of the same contract lives in ``shared/artifact_signature.py``
(``electronic_occupations`` and friends), which is also where the vocabulary is
defined so there is exactly one copy of it.

``mu`` itself is never re-derived here when the repository already knows it:
:func:`contract_from_neutrality_reference` adapts the neutrality estimate the
generalized sparse solver already publishes (``neutral_fermi_from_inertia`` /
``estimated_neutrality_reference`` in ``run_deeph_sparse_spectrum.py``).

Units: ``eps``/``mu``/``smearing_width_ev`` in eV, derivatives in eV/Ang per
unit displacement (or eV per mode, if the caller already applied the zero-point
amplitude); ``d(mu)`` comes out in the same unit as ``d(eps)``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
_SHARED_DIR = SCRIPT_DIR.parents[1] / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from artifact_signature import (  # noqa: E402
    EIGENSPACE_OCCUPATION_FIELDS,
    MU_FIXED_MU,
    MU_FIXED_N,
    MU_POLICIES,
    OCCUPATION_FUNCTIONS,
    ZERO_TEMPERATURE_STEP,
)
from epc_formalism import FORMALISM_ID_IGNORE_OVERLAP  # noqa: E402

FERMI_DIRAC = "fermi_dirac"
GAUSSIAN = "gaussian"

# Energy zeros this repository actually produces. The field is a free string —
# an external provider may name its own — but it must never be empty.
MU_CONVENTION_HAMILTONIAN_FERMI_ZERO = "exported_hamiltonian_fermi_zero"
MU_CONVENTION_ABSOLUTE = "absolute_scf_eV"

# Below this, sum(w f') is noise and d(mu) is a ratio of two zeros: a gapped
# system at low smearing has no Fermi surface to respond.
MIN_STATES_AT_MU_PER_EV = 1e-8


class EpcOccupationError(ValueError):
    """An occupation/chemical-potential rule of roadmap XII was violated."""


@dataclass(frozen=True)
class OccupationContract:
    """What one electronic artifact declares about how its states are filled.

    ``mu_ev`` is the chemical potential in force, in the declared convention;
    it may be ``None`` for ``fixed_N`` with a finite smearing, where
    :func:`resolve_chemical_potential` solves for it. ``mu_policy`` is not about
    where ``mu`` came from but about what happens when the atoms move:
    ``fixed_mu`` keeps it, ``fixed_N`` lets it follow (S7).
    """

    electron_count: float
    spin_degeneracy: int
    occupation_function: str
    smearing_width_ev: float
    mu_convention: str
    mu_policy: str = MU_FIXED_N
    mu_ev: float | None = None
    mu_source: str = "unrecorded"

    def __post_init__(self) -> None:
        if self.occupation_function not in OCCUPATION_FUNCTIONS:
            raise EpcOccupationError(
                f"occupation_function must be one of {OCCUPATION_FUNCTIONS}, "
                f"got {self.occupation_function!r}"
            )
        if self.mu_policy not in MU_POLICIES:
            raise EpcOccupationError(f"mu_policy must be one of {MU_POLICIES}, got {self.mu_policy!r}")
        if self.electron_count is None or self.electron_count <= 0:
            raise EpcOccupationError(f"electron_count must be positive, got {self.electron_count!r}")
        if int(self.spin_degeneracy) not in (1, 2):
            raise EpcOccupationError(f"spin_degeneracy must be 1 or 2, got {self.spin_degeneracy!r}")
        if not str(self.mu_convention).strip():
            raise EpcOccupationError("mu_convention must be declared (what the energy zero is)")
        zero_t = self.occupation_function == ZERO_TEMPERATURE_STEP
        if zero_t != (float(self.smearing_width_ev) == 0.0):
            raise EpcOccupationError(
                f"{self.occupation_function!r} and smearing_width_ev={self.smearing_width_ev} "
                f"disagree: only {ZERO_TEMPERATURE_STEP!r} has zero width, and it must have it"
            )
        if float(self.smearing_width_ev) < 0.0:
            raise EpcOccupationError("smearing_width_ev cannot be negative")
        if self.mu_policy == MU_FIXED_MU and self.mu_ev is None:
            raise EpcOccupationError("mu_policy 'fixed_mu' means mu is an input: give mu_ev")

    # -- what the manifests declare -----------------------------------------
    def eigenspace_fields(self) -> dict[str, Any]:
        """The subset an ``electronic_eigenspace`` node declares (no smearing)."""
        values = {
            "electron_count": float(self.electron_count),
            "spin_degeneracy": int(self.spin_degeneracy),
            "mu_convention": str(self.mu_convention),
        }
        assert set(values) == set(EIGENSPACE_OCCUPATION_FIELDS)  # kept in step with the DAG
        return values

    def artifact_fields(self) -> dict[str, Any]:
        """The full ``electronic_occupations`` node fields."""
        return {
            **self.eigenspace_fields(),
            "occupation_function": str(self.occupation_function),
            "smearing_width_ev": float(self.smearing_width_ev),
            "mu_policy": str(self.mu_policy),
            "mu_ev": None if self.mu_ev is None else float(self.mu_ev),
            "mu_source": str(self.mu_source),
        }


def contract_from_neutrality_reference(
    reference: dict[str, Any],
    *,
    occupation_function: str = ZERO_TEMPERATURE_STEP,
    smearing_width_ev: float = 0.0,
    mu_policy: str = MU_FIXED_N,
    mu_convention: str = MU_CONVENTION_HAMILTONIAN_FERMI_ZERO,
) -> OccupationContract:
    """Adapt the solver's existing neutrality estimate into a contract.

    Accepts either shape published by ``run_deeph_sparse_spectrum.py``: the
    inertia-counted one (``neutral_electrons``, ``chemical_potential_available``)
    or the evidence-only one (``expected_neutral_valence_electrons``). No second
    chemical-potential finder is introduced here — an unavailable ``mu`` stays
    unavailable and only ``fixed_N`` with a finite smearing may solve for it.
    """
    electrons = reference.get("neutral_electrons")
    if electrons is None:
        electrons = reference.get("expected_neutral_valence_electrons")
    if electrons is None:
        raise EpcOccupationError(
            "neutrality reference carries no electron count "
            f"({reference.get('limitation', 'no limitation recorded')})"
        )
    available = bool(reference.get("chemical_potential_available"))
    return OccupationContract(
        electron_count=float(electrons),
        spin_degeneracy=int(reference.get("spin_degeneracy", 2)),
        occupation_function=occupation_function,
        smearing_width_ev=float(smearing_width_ev),
        mu_convention=mu_convention,
        mu_policy=mu_policy,
        mu_ev=float(reference["energy_eV"]) if available else None,
        mu_source=str(reference.get("method", "unrecorded")),
    )


# -- occupations ------------------------------------------------------------


def _flat(eps: Any, weights: Any) -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(eps, dtype=np.float64).ravel()
    if energies.size == 0:
        raise EpcOccupationError("no eigenvalues given")
    if weights is None:
        return energies, np.ones_like(energies)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape not in (energies.shape, np.asarray(eps).shape):
        # per-k weights broadcast over that k point's bands
        w = np.broadcast_to(w.reshape(-1, 1), np.asarray(eps, dtype=np.float64).shape)
    w = w.ravel()
    if w.shape != energies.shape:
        raise EpcOccupationError(f"weights {w.shape} do not match eigenvalues {energies.shape}")
    if np.any(w < 0):
        raise EpcOccupationError("k-point weights cannot be negative")
    return energies, w


def occupations(contract: OccupationContract, eps: Any, mu: float) -> np.ndarray:
    """``f_n`` per state, in [0, 1] and spin-independent."""
    energies = np.asarray(eps, dtype=np.float64)
    if contract.occupation_function == ZERO_TEMPERATURE_STEP:
        return (energies < mu).astype(np.float64)
    x = (energies - mu) / float(contract.smearing_width_ev)
    if contract.occupation_function == FERMI_DIRAC:
        return 0.5 * (1.0 - np.tanh(0.5 * x))  # == 1/(1+exp(x)), overflow-free
    return 0.5 * erfc(x)  # gaussian smearing


def occupation_derivative(contract: OccupationContract, eps: Any, mu: float) -> np.ndarray:
    """``df_n/d(eps_n)`` in 1/eV. Non-positive, and identically zero at T = 0."""
    energies = np.asarray(eps, dtype=np.float64)
    if contract.occupation_function == ZERO_TEMPERATURE_STEP:
        return np.zeros_like(energies)
    width = float(contract.smearing_width_ev)
    x = (energies - mu) / width
    if contract.occupation_function == FERMI_DIRAC:
        f = 0.5 * (1.0 - np.tanh(0.5 * x))
        return -f * (1.0 - f) / width
    return -np.exp(-(x**2)) / (width * np.sqrt(np.pi))


def electron_count(contract: OccupationContract, eps: Any, mu: float, weights: Any = None) -> float:
    """``N(mu) = g_s sum_n w_n f_n``, the quantity ``fixed_N`` holds constant."""
    energies, w = _flat(eps, weights)
    return float(contract.spin_degeneracy * np.sum(w * occupations(contract, energies, mu)))


def resolve_chemical_potential(
    contract: OccupationContract, eps: Any = None, weights: Any = None
) -> dict[str, Any]:
    """The ``mu`` in force, and where it came from.

    ``fixed_mu`` returns the declared value. ``fixed_N`` returns the declared
    value when the neutrality reference already provided one, and otherwise
    solves ``N(mu) = electron_count`` — which needs a finite smearing, because
    at ``T = 0`` the equation is a step and the answer is the mid-gap estimate
    the sparse solver's inertia count already produces.
    """
    if contract.mu_policy == MU_FIXED_MU:
        return {"mu_ev": float(contract.mu_ev), "method": "externally_fixed", "mu_policy": MU_FIXED_MU}
    if contract.mu_ev is not None:
        return {"mu_ev": float(contract.mu_ev), "method": contract.mu_source, "mu_policy": MU_FIXED_N}
    if contract.occupation_function == ZERO_TEMPERATURE_STEP:
        raise EpcOccupationError(
            "fixed_N at zero temperature has no unique mu inside a gap: take it from the solver's "
            "neutrality reference (neutral_fermi_from_inertia) via contract_from_neutrality_reference"
        )
    if eps is None:
        raise EpcOccupationError("solving N(mu) = electron_count needs the eigenvalues")
    energies, w = _flat(eps, weights)
    margin = 20.0 * float(contract.smearing_width_ev)
    low, high = float(energies.min()) - margin, float(energies.max()) + margin

    def residual(mu: float) -> float:
        return electron_count(contract, energies, mu, w) - float(contract.electron_count)

    if residual(low) > 0 or residual(high) < 0:
        raise EpcOccupationError(
            f"the given spectrum cannot hold {contract.electron_count} electrons: "
            f"N spans [{electron_count(contract, energies, low, w)}, "
            f"{electron_count(contract, energies, high, w)}]. A shift-invert window is not a "
            f"complete spectrum — pass the full one or a declared mu_ev"
        )
    mu = float(brentq(residual, low, high, xtol=1e-12, rtol=1e-14))
    return {"mu_ev": mu, "method": "solved_from_electron_count", "mu_policy": MU_FIXED_N}


# -- the four artifacts of roadmap XII --------------------------------------


def eigenvalue_directional_derivatives(
    block: Any, contract: OccupationContract, *, units: str = "eV/Ang"
) -> dict[str, Any]:
    """Artifact 2: ``d(eps_n)`` = diag of the coupling block (generalized HF).

    No smearing enters, and ``d(mu)`` is *not* subtracted — only the declared
    energy zero matters, because a shift ``H -> H + c(R) S`` adds ``(dc) delta_mn``
    (memo section 9). The overlap-free control negative is refused: its diagonal
    is not a ``d(eps)`` at all.
    """
    formalism_id = getattr(block, "formalism_id", None)
    if formalism_id == FORMALISM_ID_IGNORE_OVERLAP:
        raise EpcOccupationError(
            "the ignore-overlap control negative has no eigenvalue derivative: "
            "diag(C^dag D_H C) is not d(eps) in a moving basis"
        )
    values = np.asarray(getattr(block, "values", block))
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise EpcOccupationError(f"expected a square block, got shape {values.shape}")
    diagonal = np.diag(values)
    imaginary = float(np.max(np.abs(np.imag(diagonal)))) if np.iscomplexobj(diagonal) else 0.0
    return {
        "artifact": "eigenvalue_directional_derivative",
        "values": np.real(diagonal).astype(np.float64),
        "max_abs_imaginary_part": imaginary,
        "mu_convention": contract.mu_convention,
        "mu_subtracted": False,
        "formalism_id": formalism_id,
        "units": units,
    }


def chemical_potential_response(
    contract: OccupationContract,
    eps: Any,
    d_eps: Any,
    *,
    weights: Any = None,
    mu: float | None = None,
    units: str = "eV/Ang",
) -> dict[str, Any]:
    """Artifact 4: ``d(mu)`` from (S7). Fixed-``N`` only.

    At fixed ``mu`` there is nothing to compute — ``d(mu) = 0`` by definition —
    and calling this is a sign the caller meant a charge response instead, so it
    raises rather than returning a zero that would then be subtracted from
    everything.
    """
    if contract.mu_policy != MU_FIXED_N:
        raise EpcOccupationError(
            f"d(mu) is identically zero under mu_policy {contract.mu_policy!r}; (S7) only applies "
            f"when N is held fixed"
        )
    energies, w = _flat(eps, weights)
    derivatives, _ = _flat(d_eps, weights)
    if derivatives.shape != energies.shape:
        raise EpcOccupationError(f"d(eps) {derivatives.shape} does not match eps {energies.shape}")
    if mu is None:
        mu = resolve_chemical_potential(contract, energies, w)["mu_ev"]
    f_prime = occupation_derivative(contract, energies, mu)
    denominator = float(np.sum(w * f_prime))
    states_at_mu = -denominator * contract.spin_degeneracy
    if states_at_mu <= MIN_STATES_AT_MU_PER_EV:
        raise EpcOccupationError(
            f"no Fermi surface to respond: sum(w f') gives {states_at_mu:.3e} states/eV at "
            f"mu = {mu:.6f} eV ({contract.occupation_function}, width "
            f"{contract.smearing_width_ev} eV). A gapped or zero-temperature artifact has no "
            f"d(mu); its eigenvalue derivatives are still valid"
        )
    return {
        "artifact": "fermi_surface_response",
        "response_quantity": "d_mu",
        "d_mu": float(np.sum(w * f_prime * derivatives) / denominator),
        "mu_ev": float(mu),
        "dos_at_mu_states_per_ev": float(states_at_mu),
        "mu_policy": contract.mu_policy,
        "mu_convention": contract.mu_convention,
        "occupation_function": contract.occupation_function,
        "smearing_width_ev": float(contract.smearing_width_ev),
        "units": units,
    }


def deformation_potentials_relative_to_mu(
    contract: OccupationContract,
    eps: Any,
    d_eps: Any,
    *,
    bands: Any = None,
    d_eps_bands: Any = None,
    weights: Any = None,
    mu: float | None = None,
    units: str = "eV/Ang",
) -> dict[str, Any]:
    """Artifact 3: ``d(eps_n) - d(mu)`` for the requested bands.

    ``eps``/``d_eps`` are the *mesh* (S7 integrates over all of it);
    ``bands``/``d_eps_bands`` are the states the deformation potential is asked
    for, defaulting to the whole mesh. The subtraction happens here and nowhere
    else — in particular never inside a ``pao_projected_mode_coupling``.
    """
    response = chemical_potential_response(
        contract, eps, d_eps, weights=weights, mu=mu, units=units
    )
    selected = np.asarray(d_eps if d_eps_bands is None else d_eps_bands, dtype=np.float64)
    return {
        "artifact": "deformation_potential_relative_to_mu",
        "values": selected - response["d_mu"],
        "bands": None if bands is None else np.asarray(bands).tolist(),
        "d_mu": response["d_mu"],
        "mu_ev": response["mu_ev"],
        "mu_policy": contract.mu_policy,
        "mu_convention": contract.mu_convention,
        "occupation_function": contract.occupation_function,
        "smearing_width_ev": float(contract.smearing_width_ev),
        "units": units,
    }


def with_smearing(
    contract: OccupationContract, occupation_function: str, smearing_width_ev: float
) -> OccupationContract:
    """Same states and electrons, different smearing — the only knob a re-smear turns."""
    return replace(
        contract, occupation_function=occupation_function, smearing_width_ev=smearing_width_ev
    )


def occupations_contract() -> dict[str, Any]:
    """The roadmap-XII contract as data, for manifests and the UI."""
    return {
        "schema": "electronic_occupations_contract_v1",
        "formula": "d(mu) = sum_n w_n f'_n d(eps_n) / sum_n w_n f'_n   (fixed N)",
        "occupation_functions": list(OCCUPATION_FUNCTIONS),
        "mu_policies": list(MU_POLICIES),
        "eigenspace_declares": list(EIGENSPACE_OCCUPATION_FIELDS),
        "artifacts": {
            "pao_projected_mode_coupling": "raw g_mn; no smearing, no d(mu)",
            "eigenvalue_directional_derivative": "diag; declared mu convention, no d(mu)",
            "deformation_potential_relative_to_mu": "d(eps) - d(mu); fixed_N only",
            "fermi_surface_response": "d(mu) itself; fixed_N only",
        },
    }
