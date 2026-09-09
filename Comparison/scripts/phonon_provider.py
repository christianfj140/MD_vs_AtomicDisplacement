#!/usr/bin/env python3
"""C17 / E-F_001-S20: the backend-independent phonon contract (Fase 5).

The EPC core needs exactly one thing from a phonon source: a mode that says
what it *is*. Not "an eigenvector from VIBRA", but

    q, omega, e, masses, normalisation, Fourier sign, atomic phase, ASR, provenance

with every one of those declared rather than assumed. This module is that
contract plus the first adapter (SIESTA FC / VIBRA); the future ML provider,
Phonopy, or a moire continuum model implement :class:`PhononProvider` and the
core does not change a line.

Canonical units, and there are no others in the returned objects:

===================  ==================================================
frequency            eV  (``omega`` as an energy, ``hbar*omega``)
displacement         Ang
mass                 amu
force constants      eV/Ang^2
q                    fractional coordinates of the reciprocal cell
===================  ==================================================

Canonical conventions (:data:`CANONICAL_CONVENTIONS`) are the ones
``docs/epc_formalismo_pao_movil.md`` section 2 already fixed for the whole
repository; every adapter converts *to* them at its boundary:

* ``fourier_sign = +1``: ``D(q) = sum_l Phi_l exp(+i q.R_l)``;
* ``atomic_phase = "cell"``: the phase carries ``R_l`` only, never
  ``R_l + tau_kappa``. A provider that folds ``tau`` into its eigenvectors
  declares ``"atom"`` and :func:`to_canonical_conventions` removes it — the
  difference is a diagonal unitary, so it is represented and converted, never
  "corrected" ad hoc (roadmap Fase 7.4);
* ``eigenvector_normalization = "mass_weighted_unit_norm"``: ``e`` diagonalises
  the mass-weighted dynamical matrix with ``sum_kappa |e_kappa|^2 = 1``.

Where the amplitude factors live, because this is the part that silently
produces wrong eV. :meth:`PhononMode.displacement_pattern` is the memo's
canonical pattern, section 8, and nothing else:

    u_{l kappa} = sqrt(hbar/(2 M_kappa omega_qnu)) * e_kappa * exp(+i q.R_l)

The amplitude is :func:`epc_formalism.zero_point_amplitude_ang`, per atom, with
the mass inside it — so the eV/Ang of ``g`` becomes eV through the same constant
the formalism uses. There is no ``1/sqrt(Nc)`` in it and none in ``g``: in the
cell gauge it cancels against the Bloch sum and reappears only as ``1/N_q`` when
an observable integrates over ``q`` (memo section 8). A provider whose
eigenvectors carry it anyway declares that and converts at its boundary;
:func:`nc_factor` is the only place the factor is written (:data:`NC_PLACEMENT`).

Two refusals, both mechanical:

* raw and ASR-corrected IFC are different artifacts. :class:`ForceConstants`
  always reports the *raw* translational residual, ``apply_asr`` returns a new
  object whose ``asr_policy`` says ``corrected``, and the residual travels into
  the mode's signature fields. An acoustic ``omega`` near zero is then a
  measurement, not a claim;
* a :class:`SyntheticTestDisplacement` is not a phonon and cannot be made into
  one. It has no frequency, eigenvector, mass or normalisation to lose, its
  artifact kind is ``synthetic_test_displacement``, and
  :func:`require_physical_phonon` is what a consumer calls to be sure
  (roadmap B7).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from artifact_signature import PHYSICAL_PHONON, SYNTHETIC_TEST_DISPLACEMENT  # noqa: E402
from epc_fourier import (  # noqa: E402
    CONVENTION_ID as QSPACE_CONVENTION_ID,
    MINUS_Q_CONVENTION,
    ATOMIC_PHASE_ATOM as _ATOMIC_PHASE_ATOM,
    ATOMIC_PHASE_CELL as _ATOMIC_PHASE_CELL,
    ATOMIC_PHASES as _ATOMIC_PHASES,
    FOURIER_SIGN_CANONICAL as _FOURIER_SIGN_CANONICAL,
    atomic_phase as _canonical_atomic_phase,
    fractional_positions,
)
from epc_formalism import (  # noqa: E402
    ZERO_POINT_CONSTANT_ANG2_AMU_EV,
    zero_point_amplitude_ang,
)
from fdf_materialization import extract_fdf_structure  # noqa: E402
from reference_provenance import canonical_sha256, file_sha256  # noqa: E402
from reference_provenance import geometry_cell_species_sha256  # noqa: E402

CONTRACT_ID = "phonon_provider_v1"

UNITS = {
    "frequency": "eV",
    "displacement": "Ang",
    "mass": "amu",
    "force_constants": "eV/Ang^2",
    "q": "fractional_reciprocal_cell",
}

# hbar in eV*Ang*sqrt(amu/eV): omega[eV] = sqrt(C * eigval(Phi/sqrt(M M'))).
_HBAR2_EV2_ANG2_AMU = ZERO_POINT_CONSTANT_ANG2_AMU_EV

CM1_TO_EV = 1.239841984e-4

# Signs and gauges are not defined here: epc_fourier is the one q-space
# convention module (D01 / E-F_001-S29) and the electronic side reads the same
# constants, which is what keeps u ~ exp(+i q.R) and H(k) ~ exp(+i k.R) in step.
FOURIER_SIGN_CANONICAL = _FOURIER_SIGN_CANONICAL
ATOMIC_PHASE_CELL = _ATOMIC_PHASE_CELL
ATOMIC_PHASE_ATOM = _ATOMIC_PHASE_ATOM
ATOMIC_PHASES = _ATOMIC_PHASES

MASS_WEIGHTED_UNIT_NORM = "mass_weighted_unit_norm"

ASR_RAW = "raw"
ASR_CORRECTED = "corrected"
ASR_POLICIES = (ASR_RAW, ASR_CORRECTED)

NC_PLACEMENT = (
    "no 1/sqrt(Nc) in the mode and none in g: in the cell gauge it cancels against "
    "the Bloch sum and enters only as 1/N_q in integrated observables (memo section 8). "
    "displacement_pattern() = sqrt(hbar/(2 M omega)) e exp(i q.R); a provider that folds "
    "the factor into its eigenvectors declares it and calls nc_factor(n_cells)."
)

SIESTA_FC_BACKEND = "siesta_fc"
VIBRA_BACKEND = "siesta_vibra"
PROVIDER_VERSION = "1"


class PhononContractError(RuntimeError):
    """A phonon object was built or used in a way the contract forbids."""


def nc_factor(n_cells: int) -> float:
    """``1/sqrt(Nc)``. The only supported way to introduce it (:data:`NC_PLACEMENT`)."""
    if n_cells < 1:
        raise PhononContractError(f"n_cells must be >= 1, got {n_cells}")
    return float(1.0 / np.sqrt(n_cells))


# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhononConventions:
    """How to read the numbers of a mode set. Never inferred, always declared."""

    fourier_sign: int = FOURIER_SIGN_CANONICAL
    atomic_phase: str = ATOMIC_PHASE_CELL
    eigenvector_normalization: str = MASS_WEIGHTED_UNIT_NORM
    nc_placement: str = NC_PLACEMENT

    def __post_init__(self) -> None:
        if self.fourier_sign not in (1, -1):
            raise PhononContractError(f"fourier_sign must be +1 or -1, got {self.fourier_sign}")
        if self.atomic_phase not in ATOMIC_PHASES:
            raise PhononContractError(
                f"atomic_phase must be one of {ATOMIC_PHASES}, got {self.atomic_phase!r}"
            )
        if self.eigenvector_normalization != MASS_WEIGHTED_UNIT_NORM:
            raise PhononContractError(
                f"only {MASS_WEIGHTED_UNIT_NORM!r} is supported; an adapter with another "
                "normalisation converts at its boundary and says so"
            )

    @property
    def is_canonical(self) -> bool:
        return (
            self.fourier_sign == FOURIER_SIGN_CANONICAL
            and self.atomic_phase == ATOMIC_PHASE_CELL
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fourier_sign": int(self.fourier_sign),
            "atomic_phase": self.atomic_phase,
            "eigenvector_normalization": self.eigenvector_normalization,
            "nc_placement": self.nc_placement,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhononConventions":
        return cls(
            fourier_sign=int(payload["fourier_sign"]),
            atomic_phase=str(payload["atomic_phase"]),
            eigenvector_normalization=str(payload["eigenvector_normalization"]),
            nc_placement=str(payload.get("nc_placement", NC_PLACEMENT)),
        )


CANONICAL_CONVENTIONS = PhononConventions()


def _atomic_phase(q_fractional: Any, cell_ang: Any, positions_ang: Any, sign: int) -> np.ndarray:
    """``exp(i * sign * q . tau_kappa)``, the whole difference between the two phases.

    Delegated to :func:`epc_fourier.atomic_phase` so the phonon gauge and the
    electronic one cannot drift apart (``q . tau = 2 pi q_frac . tau_frac``).
    """
    return _canonical_atomic_phase(
        q_fractional, fractional_positions(positions_ang, cell_ang), sign
    )


# --------------------------------------------------------------------------- #
# Force constants (raw vs ASR-corrected)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ForceConstants:
    """Real-space IFC of one FC run, in the primitive + lattice-image layout.

    ``matrix[l, kappa, alpha, kappa_prime, beta]`` is
    ``d^2 E / dR_{0 kappa alpha} dR_{l kappa' beta}`` in eV/Ang^2, and
    ``cell_images[l]`` is the integer lattice vector ``R_l`` in units of the
    primitive cell. A Gamma-only FC run is the ``n_cells == 1`` case.
    """

    matrix: np.ndarray
    cell_images: np.ndarray
    cell_ang: np.ndarray
    positions_ang: np.ndarray
    masses_amu: np.ndarray
    asr_policy: str = ASR_RAW
    asr_residual_raw_ev_ang2: float | None = None
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        images = np.asarray(self.cell_images, dtype=np.int64).reshape(-1, 3)
        n_atoms = matrix.shape[1] if matrix.ndim == 5 else -1
        if matrix.ndim != 5 or matrix.shape != (images.shape[0], n_atoms, 3, n_atoms, 3):
            raise PhononContractError(
                f"matrix must be [n_cells, na, 3, na, 3] with n_cells={images.shape[0]}, "
                f"got shape {matrix.shape}"
            )
        if self.asr_policy not in ASR_POLICIES:
            raise PhononContractError(f"asr_policy must be one of {ASR_POLICIES}")
        masses = np.asarray(self.masses_amu, dtype=np.float64).reshape(-1)
        if masses.size != n_atoms or np.any(masses <= 0.0):
            raise PhononContractError(f"masses_amu must be {n_atoms} positive amu values")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "cell_images", images)
        object.__setattr__(self, "masses_amu", masses)
        object.__setattr__(self, "cell_ang", np.asarray(self.cell_ang, dtype=np.float64))
        object.__setattr__(
            self, "positions_ang", np.asarray(self.positions_ang, dtype=np.float64).reshape(-1, 3)
        )
        if self.asr_residual_raw_ev_ang2 is None:
            object.__setattr__(self, "asr_residual_raw_ev_ang2", self.asr_residual_ev_ang2)

    @property
    def atom_count(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def cell_count(self) -> int:
        return int(self.cell_images.shape[0])

    @property
    def asr_violation(self) -> np.ndarray:
        """``sum_{l kappa'} Phi[l, kappa, alpha, kappa', beta]`` — zero by translation."""
        return self.matrix.sum(axis=(0, 3))

    @property
    def asr_residual_ev_ang2(self) -> float:
        """The largest translational violation of *this* matrix, in eV/Ang^2."""
        return float(np.abs(self.asr_violation).max())

    def apply_asr(self) -> "ForceConstants":
        """Put the violation back on the on-site block; the raw residual survives."""
        if self.asr_policy == ASR_CORRECTED:
            return self
        zero = int(np.argmin(np.abs(self.cell_images).sum(axis=1)))
        if np.any(self.cell_images[zero] != 0):
            raise PhononContractError("cell_images has no R = 0 image to carry the ASR correction")
        matrix = self.matrix.copy()
        violation = self.asr_violation
        for kappa in range(self.atom_count):
            matrix[zero, kappa, :, kappa, :] -= violation[kappa]
        return replace(
            self,
            matrix=matrix,
            asr_policy=ASR_CORRECTED,
            asr_residual_raw_ev_ang2=self.asr_residual_ev_ang2,
        )

    def dynamical_matrix(
        self,
        q_fractional: Sequence[float] = (0.0, 0.0, 0.0),
        conventions: PhononConventions = CANONICAL_CONVENTIONS,
    ) -> np.ndarray:
        """``D(q)`` in eV/(Ang^2 amu), ``[3na, 3na]``, in the given conventions."""
        q = np.asarray(q_fractional, dtype=np.float64).reshape(3)
        phases = np.exp(2j * np.pi * conventions.fourier_sign * (self.cell_images @ q))
        n_atoms = self.atom_count
        matrix = np.tensordot(phases, self.matrix, axes=(0, 0)).reshape(3 * n_atoms, 3 * n_atoms)
        if conventions.atomic_phase == ATOMIC_PHASE_ATOM:
            tau = _atomic_phase(q, self.cell_ang, self.positions_ang, conventions.fourier_sign)
            unitary = np.repeat(tau, 3)
            matrix = unitary.conj()[:, None] * matrix * unitary[None, :]
        inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(self.masses_amu), 3)
        return inverse_sqrt_mass[:, None] * matrix * inverse_sqrt_mass[None, :]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "atom_count": self.atom_count,
            "cell_images": self.cell_images.tolist(),
            "asr_policy": self.asr_policy,
            "asr_residual_ev_ang2": self.asr_residual_ev_ang2,
            "asr_residual_raw_ev_ang2": self.asr_residual_raw_ev_ang2,
            "units": UNITS["force_constants"],
            "provenance": dict(self.provenance or {}),
        }


def ifc_shells(force_constants: ForceConstants) -> dict[str, Any]:
    """Distance-resolved size of the IFC: the tail of the auxiliary supercell.

    One shell per distinct ``|R_l + tau_kappa' - tau_kappa|``, each carrying the
    largest element and the Frobenius norm of its ``3x3`` blocks, plus what that
    shell contributes to ``D(Gamma) = sum_l Phi_l``. ``tail_ratio`` is the
    outermost shell measured against the on-site one: it is the number that says
    whether widening the FC supercell would still change anything.
    """
    matrix = force_constants.matrix
    positions = force_constants.positions_ang
    lattice = force_constants.cell_images @ force_constants.cell_ang  # [n_cells, 3]
    n_atoms = force_constants.atom_count
    gamma = matrix.sum(axis=0)
    gamma_norm = float(np.linalg.norm(gamma))

    by_distance: dict[float, list[tuple[int, int, int]]] = {}
    for cell in range(force_constants.cell_count):
        for kappa in range(n_atoms):
            for kappa_prime in range(n_atoms):
                separation = positions[kappa_prime] + lattice[cell] - positions[kappa]
                key = round(float(np.linalg.norm(separation)), 6)
                by_distance.setdefault(key, []).append((cell, kappa, kappa_prime))

    shells: list[dict[str, Any]] = []
    for distance in sorted(by_distance):
        blocks = np.array(
            [matrix[cell, kappa, :, kappa_prime, :] for cell, kappa, kappa_prime in by_distance[distance]]
        )
        contribution = np.array(
            [gamma[kappa, :, kappa_prime, :] for _, kappa, kappa_prime in by_distance[distance]]
        )
        shells.append(
            {
                "distance_ang": distance,
                "block_count": int(blocks.shape[0]),
                "max_abs_ev_ang2": float(np.abs(blocks).max()),
                "frobenius_ev_ang2": float(np.linalg.norm(blocks)),
                # A shell only contributes to Gamma through the images it holds;
                # at n_cells = 1 every image is already folded into that block.
                "gamma_contribution_fraction": (
                    float(np.linalg.norm(contribution) / gamma_norm) if gamma_norm else 0.0
                ),
            }
        )
    onsite = shells[0]["max_abs_ev_ang2"]
    return {
        "asr_policy": force_constants.asr_policy,
        "cell_count": force_constants.cell_count,
        "shell_count": len(shells),
        "max_distance_ang": shells[-1]["distance_ang"],
        "shells": shells,
        "tail_ratio": float(shells[-1]["max_abs_ev_ang2"] / onsite) if onsite else float("inf"),
        "tail_policy": (
            "tail_ratio = max|Phi| of the outermost shell / max|Phi| of the on-site shell; "
            "an FC run of n_cells = 1 has no tail to measure because every image is folded "
            "into the single block"
        ),
    }


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhononMode:
    """One branch at one q: everything needed to displace atoms, nothing else."""

    q_fractional: np.ndarray
    branch: int
    frequency_ev: float
    eigenvector: np.ndarray  # [na, 3], mass weighted, unit norm
    masses_amu: np.ndarray
    cell_ang: np.ndarray
    positions_ang: np.ndarray
    conventions: PhononConventions

    @property
    def is_imaginary(self) -> bool:
        """A negative ``frequency_ev`` is the reported sign of ``omega^2 < 0``."""
        return self.frequency_ev < 0.0

    @property
    def zero_point_amplitude_ang(self) -> np.ndarray:
        """``sqrt(hbar / (2 M_kappa omega))`` per atom, in Ang (memo section 8)."""
        if self.frequency_ev <= 0.0:
            raise PhononContractError(
                f"branch {self.branch} has omega = {self.frequency_ev:.3e} eV: the zero-point "
                "amplitude of an acoustic (omega -> 0) or imaginary mode diverges. Displace it "
                "with an explicit amplitude instead, and say so"
            )
        return np.array(
            [zero_point_amplitude_ang(float(m), self.frequency_ev) for m in self.masses_amu]
        )

    def displacement_pattern(self, cell_image: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Complex ``[na, 3]`` displacement in Ang, ``Nc``-free (:data:`NC_PLACEMENT`).

        ``u_kappa = sqrt(hbar/(2 M_kappa omega)) e_kappa exp(i s q.R_l)``, with the
        atomic phase included only if the conventions say the eigenvector does
        not already carry it.
        """
        image = np.asarray(cell_image, dtype=np.float64).reshape(3)
        phase = np.exp(2j * np.pi * self.conventions.fourier_sign * float(self.q_fractional @ image))
        pattern = self.zero_point_amplitude_ang[:, None] * self.eigenvector * phase
        if self.conventions.atomic_phase == ATOMIC_PHASE_ATOM:
            tau = _atomic_phase(
                self.q_fractional, self.cell_ang, self.positions_ang, self.conventions.fourier_sign
            )
            pattern = pattern * tau[:, None]
        return pattern


@dataclass(frozen=True)
class PhononModeSet:
    """Every branch at one q, plus the provenance that makes them citable.

    This is the *only* object the EPC core consumes. ``artifact_kind`` is
    ``physical_phonon``, and :meth:`physical_phonon_fields` produces exactly the
    fields ``artifact_signature.epc_artifact_node`` declares for that kind.
    """

    q_fractional: np.ndarray
    frequencies_ev: np.ndarray
    eigenvectors: np.ndarray  # [n_modes, na, 3]
    masses_amu: np.ndarray
    cell_ang: np.ndarray
    positions_ang: np.ndarray
    conventions: PhononConventions
    provider: str
    provider_version: str
    force_source: dict[str, Any]
    primitive_mapping: dict[str, Any]
    fc_range: dict[str, Any]
    asr_policy: str
    asr_residual_ev_ang2: float
    asr_residual_raw_ev_ang2: float
    geometry_signature: str = ""
    provenance: dict[str, Any] | None = None

    artifact_kind = PHYSICAL_PHONON

    def __post_init__(self) -> None:
        eigenvectors = np.asarray(self.eigenvectors, dtype=np.complex128)
        frequencies = np.asarray(self.frequencies_ev, dtype=np.float64).reshape(-1)
        if eigenvectors.ndim != 3 or eigenvectors.shape[0] != frequencies.size:
            raise PhononContractError(
                f"eigenvectors must be [n_modes, na, 3] with n_modes={frequencies.size}, "
                f"got {eigenvectors.shape}"
            )
        norms = np.linalg.norm(eigenvectors.reshape(frequencies.size, -1), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-8):
            raise PhononContractError(
                f"{MASS_WEIGHTED_UNIT_NORM} means ||e|| = 1; got norms in "
                f"[{norms.min():.6g}, {norms.max():.6g}]"
            )
        if self.asr_policy not in ASR_POLICIES:
            raise PhononContractError(f"asr_policy must be one of {ASR_POLICIES}")
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "frequencies_ev", frequencies)
        object.__setattr__(
            self, "q_fractional", np.asarray(self.q_fractional, dtype=np.float64).reshape(3)
        )
        object.__setattr__(self, "masses_amu", np.asarray(self.masses_amu, dtype=np.float64))
        object.__setattr__(self, "cell_ang", np.asarray(self.cell_ang, dtype=np.float64))
        object.__setattr__(
            self, "positions_ang", np.asarray(self.positions_ang, dtype=np.float64).reshape(-1, 3)
        )

    @property
    def mode_count(self) -> int:
        return int(self.frequencies_ev.size)

    def mode(self, branch: int) -> PhononMode:
        return PhononMode(
            q_fractional=self.q_fractional,
            branch=int(branch),
            frequency_ev=float(self.frequencies_ev[branch]),
            eigenvector=self.eigenvectors[branch],
            masses_amu=self.masses_amu,
            cell_ang=self.cell_ang,
            positions_ang=self.positions_ang,
            conventions=self.conventions,
        )

    @property
    def eigenvector_sha256(self) -> str:
        # ponytail: the +0.0 is not decoration — round() keeps -0.0, JSON keeps
        # it too, and repr(-0.0) != repr(0.0) would make a round trip change the
        # signature and invalidate every cached g built on this mode.
        rounded = np.round(self.eigenvectors, 12) + 0.0
        return canonical_sha256({"real": rounded.real.tolist(), "imag": rounded.imag.tolist()})

    def physical_phonon_fields(self) -> dict[str, Any]:
        """The ``physical_phonon`` node fields, exactly as the EPC DAG declares them."""
        return {
            "provider": self.provider,
            "provider_version": self.provider_version,
            "force_source": dict(self.force_source),
            "primitive_mapping": dict(self.primitive_mapping),
            "fc_range": dict(self.fc_range),
            "q": self.q_fractional.tolist(),
            "frequency_ev": self.frequencies_ev.tolist(),
            "eigenvector_sha256": self.eigenvector_sha256,
            "masses": self.masses_amu.tolist(),
            "normalization": {
                "eigenvector": self.conventions.eigenvector_normalization,
                "nc_placement": self.conventions.nc_placement,
                "zero_point_amplitude": "sqrt(hbar/(2 M omega)) in Ang",
            },
            "asr_policy": {
                "policy": self.asr_policy,
                "residual_ev_ang2": self.asr_residual_ev_ang2,
                "raw_residual_ev_ang2": self.asr_residual_raw_ev_ang2,
            },
            "fourier_convention": {
                "sign": int(self.conventions.fourier_sign),
                "atomic_phase": self.conventions.atomic_phase,
                "contract_id": CONTRACT_ID,
                # The electronic side signs the same id, so "same convention" is
                # checkable between a phonon and a D_H (S29).
                "convention_id": QSPACE_CONVENTION_ID,
                "minus_q": MINUS_Q_CONVENTION,
            },
            "units": dict(UNITS),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe round trip: masses, normalisation, q and phases all survive."""
        return {
            "contract_id": CONTRACT_ID,
            "artifact_kind": self.artifact_kind,
            "q_fractional": self.q_fractional.tolist(),
            "frequencies_ev": self.frequencies_ev.tolist(),
            "eigenvectors_real": self.eigenvectors.real.tolist(),
            "eigenvectors_imag": self.eigenvectors.imag.tolist(),
            "masses_amu": self.masses_amu.tolist(),
            "cell_ang": self.cell_ang.tolist(),
            "positions_ang": self.positions_ang.tolist(),
            "conventions": self.conventions.to_dict(),
            "provider": self.provider,
            "provider_version": self.provider_version,
            "force_source": dict(self.force_source),
            "primitive_mapping": dict(self.primitive_mapping),
            "fc_range": dict(self.fc_range),
            "asr_policy": self.asr_policy,
            "asr_residual_ev_ang2": self.asr_residual_ev_ang2,
            "asr_residual_raw_ev_ang2": self.asr_residual_raw_ev_ang2,
            "geometry_signature": self.geometry_signature,
            "provenance": dict(self.provenance or {}),
            "units": dict(UNITS),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhononModeSet":
        kind = payload.get("artifact_kind")
        if kind != PHYSICAL_PHONON:
            raise PhononContractError(
                f"payload declares artifact_kind {kind!r}; a {PHYSICAL_PHONON} cannot be "
                f"read out of it (roadmap B7)"
            )
        eigenvectors = np.asarray(payload["eigenvectors_real"], dtype=np.float64) + 1j * np.asarray(
            payload["eigenvectors_imag"], dtype=np.float64
        )
        return cls(
            q_fractional=payload["q_fractional"],
            frequencies_ev=payload["frequencies_ev"],
            eigenvectors=eigenvectors,
            masses_amu=payload["masses_amu"],
            cell_ang=payload["cell_ang"],
            positions_ang=payload["positions_ang"],
            conventions=PhononConventions.from_dict(payload["conventions"]),
            provider=payload["provider"],
            provider_version=payload["provider_version"],
            force_source=payload["force_source"],
            primitive_mapping=payload["primitive_mapping"],
            fc_range=payload["fc_range"],
            asr_policy=payload["asr_policy"],
            asr_residual_ev_ang2=payload["asr_residual_ev_ang2"],
            asr_residual_raw_ev_ang2=payload["asr_residual_raw_ev_ang2"],
            geometry_signature=payload.get("geometry_signature", ""),
            provenance=payload.get("provenance"),
        )


@dataclass(frozen=True)
class SyntheticTestDisplacement:
    """A benchmark displacement. Not a phonon, and it cannot become one.

    No frequency, no eigenvector, no masses, no ASR — the fields a physical
    ``g_{mn nu}`` needs are absent by construction, not by policy (roadmap 10b).
    """

    label: str
    displacement_ang: np.ndarray
    normalization: str = "unit_frobenius_norm"
    provenance: dict[str, Any] | None = None

    artifact_kind = SYNTHETIC_TEST_DISPLACEMENT

    def __post_init__(self) -> None:
        displacement = np.asarray(self.displacement_ang, dtype=np.float64).reshape(-1, 3)
        object.__setattr__(self, "displacement_ang", displacement)

    @property
    def displacement_sha256(self) -> str:
        return canonical_sha256((np.round(self.displacement_ang, 12) + 0.0).tolist())

    def artifact_fields(self) -> dict[str, Any]:
        """The ``synthetic_test_displacement`` node fields. A different kind, on purpose."""
        return {
            "label": self.label,
            "displacement_sha256": self.displacement_sha256,
            "normalization": self.normalization,
            "units": {"displacement": UNITS["displacement"]},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": CONTRACT_ID,
            "artifact_kind": self.artifact_kind,
            **self.artifact_fields(),
            "displacement_ang": self.displacement_ang.tolist(),
            "provenance": dict(self.provenance or {}),
        }


def require_physical_phonon(mode_set: Any) -> PhononModeSet:
    """Gate for every consumer of a mode: a synthetic displacement stops here."""
    if isinstance(mode_set, PhononModeSet):
        return mode_set
    kind = getattr(mode_set, "artifact_kind", type(mode_set).__name__)
    raise PhononContractError(
        f"a {PHYSICAL_PHONON} is required and {kind!r} is not one: it carries no omega, "
        f"eigenvector, mass or normalisation, so no g_mn,nu may be built from it (roadmap B7)"
    )


def to_canonical_conventions(mode_set: PhononModeSet) -> PhononModeSet:
    """Convert an adapter's conventions to the repository canon, reversibly.

    Fourier sign: with real IFC, ``D^(-)(q) = conj(D^(+)(q))``, so flipping the
    sign conjugates the eigenvectors. Atomic phase: ``e^cell_kappa =
    e^atom_kappa exp(+i s q.tau_kappa)``. Both are exact and involutive — the
    round trip is machine precision, which is what the adapter test asserts.
    """
    conventions = mode_set.conventions
    if conventions.is_canonical:
        return mode_set
    eigenvectors = mode_set.eigenvectors
    if conventions.fourier_sign != FOURIER_SIGN_CANONICAL:
        eigenvectors = eigenvectors.conj()
    if conventions.atomic_phase == ATOMIC_PHASE_ATOM:
        tau = _atomic_phase(
            mode_set.q_fractional, mode_set.cell_ang, mode_set.positions_ang, FOURIER_SIGN_CANONICAL
        )
        eigenvectors = eigenvectors * tau[:, None]
    return replace(
        mode_set,
        eigenvectors=eigenvectors,
        conventions=PhononConventions(
            fourier_sign=FOURIER_SIGN_CANONICAL,
            atomic_phase=ATOMIC_PHASE_CELL,
            eigenvector_normalization=conventions.eigenvector_normalization,
            nc_placement=conventions.nc_placement,
        ),
    )


def modes_from_force_constants(
    force_constants: ForceConstants,
    q_fractional: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    provider: str,
    provider_version: str = PROVIDER_VERSION,
    force_source: dict[str, Any] | None = None,
    primitive_mapping: dict[str, Any] | None = None,
    fc_range: dict[str, Any] | None = None,
    geometry_signature: str = "",
    conventions: PhononConventions = CANONICAL_CONVENTIONS,
) -> PhononModeSet:
    """Diagonalise ``D(q)`` into a mode set. ``omega < 0`` reports ``omega^2 < 0``."""
    dynamical = force_constants.dynamical_matrix(q_fractional, conventions)
    hermiticity = float(np.abs(dynamical - dynamical.conj().T).max())
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (dynamical + dynamical.conj().T))
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues) * _HBAR2_EV2_ANG2_AMU)
    n_atoms = force_constants.atom_count
    return PhononModeSet(
        q_fractional=q_fractional,
        frequencies_ev=frequencies,
        eigenvectors=eigenvectors.T.reshape(3 * n_atoms, n_atoms, 3),
        masses_amu=force_constants.masses_amu,
        cell_ang=force_constants.cell_ang,
        positions_ang=force_constants.positions_ang,
        conventions=conventions,
        provider=provider,
        provider_version=provider_version,
        force_source=force_source or dict(force_constants.provenance or {}),
        primitive_mapping=primitive_mapping
        or {
            "atom_count": n_atoms,
            "positions_ang": force_constants.positions_ang.tolist(),
            "cell_ang": force_constants.cell_ang.tolist(),
        },
        fc_range=fc_range
        or {
            "cell_images": force_constants.cell_images.tolist(),
            "cell_count": force_constants.cell_count,
        },
        asr_policy=force_constants.asr_policy,
        asr_residual_ev_ang2=force_constants.asr_residual_ev_ang2,
        asr_residual_raw_ev_ang2=float(force_constants.asr_residual_raw_ev_ang2),
        geometry_signature=geometry_signature,
        provenance={"dynamical_matrix_hermiticity_ev_ang2_amu": hermiticity},
    )


def frequency_resolutions_ev(frequencies_ev: Any, noise_ev_ang2_amu: float) -> np.ndarray:
    """How finely two branches can be told apart, from the IFC's own noise floor.

    A perturbation ``dlambda`` of the dynamical matrix moves a frequency by
    ``sqrt(omega^2 + C dlambda) - omega`` exactly: ``C dlambda / (2 omega)`` for a
    resolved branch, ``sqrt(C dlambda)`` for one sitting at zero. That is where
    the degeneracy tolerance comes from — never a hand-picked cm^-1.
    """
    omega = np.abs(np.asarray(frequencies_ev, dtype=np.float64).reshape(-1))
    shift = _HBAR2_EV2_ANG2_AMU * max(float(noise_ev_ang2_amu), 0.0)
    return np.sqrt(omega**2 + shift) - omega


def mode_sector_report(
    mode_set: PhononModeSet,
    *,
    noise_ev_ang2_amu: float | None = None,
    separation_factor: float = 1.0,
) -> dict[str, Any]:
    """Which branches are acoustic and which are the degenerate optical sector.

    Measured, not imposed. The acoustic set is whichever branches live in the
    mass-weighted uniform-translation subspace — ``t_alpha[kappa] =
    sqrt(M_kappa) e_alpha``, the mass-weighted image of a rigid shift — reported
    as a projection weight rather than a branch index. The degeneracies come
    from ``epc_subspaces.near_degenerate_clusters`` at the resolution of
    :func:`frequency_resolutions_ev`, whose noise floor defaults to the *raw*
    translational residual of the IFC: a matrix that violates a rule it must
    satisfy exactly is wrong by at least that much. The plane is the one the
    lattice defines (the normal of ``a1 x a2``), so "in-plane E2g" is a
    measurement of the eigenvector, not a label attached to it.
    """
    # ponytail: imported here so the contract objects stay importable without
    # the solver stack epc_subspaces brings in.
    from epc_subspaces import cluster_groups, near_degenerate_clusters

    masses = mode_set.masses_amu
    eigenvectors = mode_set.eigenvectors  # [n_modes, na, 3]
    frequencies = mode_set.frequencies_ev
    noise = (
        float(mode_set.asr_residual_raw_ev_ang2) / float(masses.min())
        if noise_ev_ang2_amu is None
        else float(noise_ev_ang2_amu)
    )
    resolutions = frequency_resolutions_ev(frequencies, noise)

    translations = np.zeros((3, masses.size, 3))
    for axis in range(3):
        translations[axis, :, axis] = np.sqrt(masses)
        translations[axis] /= np.linalg.norm(translations[axis])
    overlaps = np.tensordot(eigenvectors.conj(), translations, axes=([1, 2], [1, 2]))
    translation_weight = np.abs(overlaps) ** 2 @ np.ones(3)

    cell = mode_set.cell_ang
    normal = np.cross(cell[0], cell[1])
    normal = normal / np.linalg.norm(normal)
    out_of_plane = np.abs(eigenvectors @ normal) ** 2 @ np.ones(masses.size)

    labels = near_degenerate_clusters(frequencies, resolutions, separation_factor=separation_factor)
    groups = cluster_groups(labels)
    clusters = [
        {
            "branches": list(group),
            "size": len(group),
            "mean_frequency_ev": float(np.mean(frequencies[group])),
            "mean_frequency_cm1": float(np.mean(frequencies[group])) / CM1_TO_EV,
            "frequency_spread_ev": float(frequencies[group].max() - frequencies[group].min()),
            "resolution_ev": float(np.max(resolutions[group])),
            "translation_weight": float(np.mean(translation_weight[group])),
            "out_of_plane_weight": float(np.mean(out_of_plane[group])),
        }
        for group in groups
    ]

    acoustic = [index for index in range(frequencies.size) if translation_weight[index] >= 0.5]
    optical = [index for index in range(frequencies.size) if index not in acoustic]
    doublets = [
        cluster
        for cluster in clusters
        if cluster["size"] == 2
        and cluster["translation_weight"] < 0.5
        and cluster["out_of_plane_weight"] < 0.1
    ]
    e2g = max(doublets, key=lambda cluster: cluster["mean_frequency_ev"]) if doublets else None
    return {
        "noise_floor_ev_ang2_amu": noise,
        "separation_factor": float(separation_factor),
        "branches": [
            {
                "branch": index,
                "frequency_ev": float(frequencies[index]),
                "frequency_cm1": float(frequencies[index]) / CM1_TO_EV,
                "resolution_ev": float(resolutions[index]),
                "translation_weight": float(translation_weight[index]),
                "out_of_plane_weight": float(out_of_plane[index]),
                "cluster": int(labels[index]),
            }
            for index in range(frequencies.size)
        ],
        "clusters": clusters,
        "acoustic_branches": acoustic,
        "optical_branches": optical,
        "in_plane_optical_doublets": doublets,
        "e2g_candidate": e2g,
        "identification": {
            # A sector is only "identified" if the projection separates it; the
            # numbers are published either way.
            "acoustic_count": len(acoustic),
            "acoustic_expected": 3,
            "min_acoustic_translation_weight": float(min(translation_weight[acoustic], default=0.0)),
            "max_optical_translation_weight": float(max(translation_weight[optical], default=0.0)),
            "acoustic_unambiguous": bool(
                len(acoustic) == 3
                and min(translation_weight[acoustic], default=0.0) > 0.9
                and max(translation_weight[optical], default=1.0) < 0.1
            ),
            "e2g_identified": e2g is not None,
            "e2g_degeneracy_resolved_as_one_cluster": bool(
                e2g is not None and e2g["frequency_spread_ev"] <= e2g["resolution_ev"]
            ),
            "policy": (
                "acoustic = projection weight >= 0.5 on the mass-weighted uniform translations; "
                "E2g = the highest in-plane, non-translational cluster of size 2 at the "
                "resolution of the IFC noise floor. No eigenvector is ever imposed"
            ),
        },
    }


class PhononProvider(Protocol):
    """What the EPC core knows about a phonon source. Nothing backend specific."""

    backend: str
    provider_version: str

    def modes(self, q_fractional: Sequence[float] = (0.0, 0.0, 0.0)) -> PhononModeSet:
        """Every branch at ``q``, in the canonical conventions and units."""


# --------------------------------------------------------------------------- #
# Adapter: SIESTA FC run (and the VIBRA output of the same run)
# --------------------------------------------------------------------------- #
# Everything SIESTA/VIBRA specific lives below this line. The core imports the
# objects above and never a parser.

_FC_HEADER = re.compile(r"n_atoms,\s*displacement\s*\[Ang\]\s*:\s*(\d+)\s+(\S+)", re.IGNORECASE)


def read_siesta_fc(fc_path: Path) -> dict[str, Any]:
    """Parse a SIESTA ``.FC``: the raw blocks, the displacement, the two signs.

    Layout, one line per atom of the FC run, ``fx fy fz`` in eV/Ang^2:
    ``[displaced atom][x, y, z][-delta, +delta][atom]``. SIESTA writes
    ``-F/delta`` with the *signed* delta, so the central difference of the pair
    is their mean — which is also what makes the resulting matrix symmetric and
    the acoustic sum rule nearly satisfied (both are asserted downstream).
    """
    fc_path = Path(fc_path)
    lines = [line for line in fc_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = _FC_HEADER.search(lines[0])
    if header is None:
        raise PhononContractError(f"{fc_path}: first line is not a SIESTA FC header: {lines[0]!r}")
    row_count = int(header.group(1))
    displacement_ang = float(header.group(2).replace("D", "E"))
    rows = np.array([[float(value) for value in line.split()] for line in lines[1:]])
    if rows.shape[1] != 3 or rows.shape[0] % (6 * row_count):
        raise PhononContractError(
            f"{fc_path}: {rows.shape[0]}x{rows.shape[1]} data rows are not "
            f"6 * n_displaced * {row_count} triplets"
        )
    displaced_count = rows.shape[0] // (6 * row_count)
    blocks = rows.reshape(displaced_count, 3, 2, row_count, 3)
    return {
        "path": str(fc_path),
        "sha256": file_sha256(fc_path),
        "displacement_ang": displacement_ang,
        "displaced_atom_count": displaced_count,
        "row_atom_count": row_count,
        "signed_blocks": blocks,
        "matrix_ev_ang2": blocks.mean(axis=2),  # [displaced, 3, atom, 3]
    }


def _masses_amu(atomic_numbers: Sequence[int]) -> np.ndarray:
    """Atomic masses in amu. ``sisl`` is already a dependency of this repository."""
    # ponytail: imported here, not at module scope — the contract objects above
    # must be usable (and testable) without paying for the sisl import.
    import sisl

    masses = []
    for number in atomic_numbers:
        if number <= 0:
            raise PhononContractError(
                f"atomic number {number} is a ghost/floating basis species: it has no mass and "
                "cannot carry a phonon displacement"
            )
        masses.append(float(sisl.Atom(int(number)).mass))
    return np.array(masses, dtype=np.float64)


def masses_amu_from_fdf(fdf_path: Path) -> np.ndarray:
    """Masses in amu of the atoms an fdf declares, in its own atom order."""
    structure = extract_fdf_structure(Path(fdf_path))
    species = {item.index: item.atomic_number for item in structure.species}
    return _masses_amu([species[index] for index in structure.atom_species])


def force_constants_from_siesta_run(
    fc_path: Path,
    fdf_path: Path,
    *,
    masses_amu: Sequence[float] | None = None,
    cell_images: Sequence[Sequence[int]] | None = None,
) -> ForceConstants:
    """Build :class:`ForceConstants` from an FC run's ``.FC`` plus its RUN.fdf.

    Only the Gamma-only case (the FC run cell *is* the primitive cell) is
    assembled automatically. A ``fcbuild`` supercell run needs the image of each
    of its atoms, which is an ordering of that tool and not of this contract:
    pass ``cell_images``, or build :class:`ForceConstants` directly.
    """
    fc_path, fdf_path = Path(fc_path), Path(fdf_path)
    parsed = read_siesta_fc(fc_path)
    structure = extract_fdf_structure(fdf_path)
    n_atoms = parsed["row_atom_count"]
    if structure.atom_count != n_atoms or parsed["displaced_atom_count"] != n_atoms:
        raise PhononContractError(
            f"{fc_path.name}: {parsed['displaced_atom_count']} displaced atoms and {n_atoms} force "
            f"rows against {structure.atom_count} atoms in {fdf_path.name}. A supercell FC run "
            "needs an explicit cell_images mapping (fcbuild ordering is not encoded here)"
        )
    if cell_images is None:
        cell_images = [[0, 0, 0]]
    elif len(cell_images) != 1:
        raise PhononContractError(
            "cell_images with more than one image needs force rows per image; build "
            "ForceConstants directly from a mapped supercell FC run"
        )
    if masses_amu is None:
        masses_amu = masses_amu_from_fdf(fdf_path)
    return ForceConstants(
        matrix=parsed["matrix_ev_ang2"].reshape(1, n_atoms, 3, n_atoms, 3),
        cell_images=cell_images,
        cell_ang=np.array(structure.lattice_vectors_ang, dtype=np.float64),
        positions_ang=np.array(structure.positions_ang, dtype=np.float64),
        masses_amu=masses_amu,
        asr_policy=ASR_RAW,
        provenance={
            "backend": SIESTA_FC_BACKEND,
            "fc_file": parsed["path"],
            "fc_sha256": parsed["sha256"],
            "fc_displacement_ang": parsed["displacement_ang"],
            "fdf": str(fdf_path),
            "fdf_sha256": file_sha256(fdf_path),
            "geometry_signature": geometry_cell_species_sha256(fdf_path),
        },
    )


def supercell_images(repeats: Sequence[int]) -> np.ndarray:
    """Lattice images of an auxiliary FC supercell, ``R = 0`` first.

    Odd repeats only: ``2n+1`` cells along an axis is the only replication whose
    minimum-image set is symmetric around the displaced cell, and an IFC tail
    measured on a lopsided shell is not a tail. The order is this module's, not
    ``fcbuild``'s — :func:`supercell_geometry` writes the SIESTA coordinates in
    exactly it, so the FC rows and ``cell_images`` cannot drift apart.
    """
    counts = [int(value) for value in repeats]
    if len(counts) != 3 or any(count < 1 or count % 2 == 0 for count in counts):
        raise PhononContractError(f"repeats must be three odd positive integers, got {repeats!r}")
    images = [
        (first, second, third)
        for first in range(-(counts[0] // 2), counts[0] // 2 + 1)
        for second in range(-(counts[1] // 2), counts[1] // 2 + 1)
        for third in range(-(counts[2] // 2), counts[2] // 2 + 1)
    ]
    images.sort(key=lambda image: (max(abs(value) for value in image), image))
    return np.array(images, dtype=np.int64)


def supercell_geometry(
    positions_ang: Any, cell_ang: Any, images: Any
) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, lattice)`` of the auxiliary supercell, image-major.

    Atom ``l * na + kappa`` is atom ``kappa`` of image ``l``, so the ``na`` atoms
    of the unit cell come first and ``FC.First 1`` / ``FC.Last na`` displaces
    exactly them.
    """
    positions = np.asarray(positions_ang, dtype=np.float64).reshape(-1, 3)
    cell = np.asarray(cell_ang, dtype=np.float64).reshape(3, 3)
    images = np.asarray(images, dtype=np.int64).reshape(-1, 3)
    supercell = (positions[None, :, :] + (images @ cell)[:, None, :]).reshape(-1, 3)
    repeats = images.max(axis=0) - images.min(axis=0) + 1
    return supercell, repeats[:, None] * cell


def force_constants_from_siesta_supercell_run(
    fc_path: Path,
    *,
    images: Any,
    cell_ang: Any,
    positions_ang: Any,
    masses_amu: Sequence[float],
    provenance: dict[str, Any] | None = None,
) -> ForceConstants:
    """Assemble ``Phi[l, kappa, alpha, kappa', beta]`` from a supercell FC run.

    The ``.FC`` holds ``[displaced unit atom][xyz][-+][supercell atom][xyz]``, and
    the supercell atom index is ``l * na + kappa'`` by construction of
    :func:`supercell_geometry`, so the mapping is a reshape rather than a guess
    about anyone's atom ordering. ``cell_ang``/``positions_ang`` stay those of
    the *unit* cell: the FC supercell is the range of the interaction, not a new
    primitive cell.
    """
    parsed = read_siesta_fc(Path(fc_path))
    images = np.asarray(images, dtype=np.int64).reshape(-1, 3)
    n_atoms = len(masses_amu)
    expected_rows = images.shape[0] * n_atoms
    if parsed["displaced_atom_count"] != n_atoms or parsed["row_atom_count"] != expected_rows:
        raise PhononContractError(
            f"{fc_path}: {parsed['displaced_atom_count']} displaced atoms and "
            f"{parsed['row_atom_count']} force rows; a {images.shape[0]}-image supercell run of "
            f"{n_atoms} unit-cell atoms writes {n_atoms} and {expected_rows}"
        )
    matrix = parsed["matrix_ev_ang2"].reshape(n_atoms, 3, images.shape[0], n_atoms, 3)
    return ForceConstants(
        matrix=matrix.transpose(2, 0, 1, 3, 4),
        cell_images=images,
        cell_ang=cell_ang,
        positions_ang=positions_ang,
        masses_amu=masses_amu,
        asr_policy=ASR_RAW,
        provenance={
            "backend": SIESTA_FC_BACKEND,
            "fc_file": parsed["path"],
            "fc_sha256": parsed["sha256"],
            "fc_displacement_ang": parsed["displacement_ang"],
            "supercell_repeats": (images.max(axis=0) - images.min(axis=0) + 1).tolist(),
            **dict(provenance or {}),
        },
    )


@dataclass(frozen=True)
class SiestaFcPhononProvider:
    """First backend: the ``.FC`` of a SIESTA FC run, the same file VIBRA reads.

    Diagonalising it here instead of trusting ``vibra`` is what makes the ASR
    policy explicit: VIBRA reports neither the raw translational residual nor
    which of the two IFC a frequency came from.
    """

    force_constants: ForceConstants
    apply_asr: bool = True
    geometry_signature: str = ""

    backend = SIESTA_FC_BACKEND
    provider_version = PROVIDER_VERSION

    @classmethod
    def from_run_dir(cls, run_dir: Path, *, apply_asr: bool = True, **kwargs: Any) -> "SiestaFcPhononProvider":
        run_dir = Path(run_dir)
        fc_files = sorted(run_dir.glob("*.FC"))
        fdf_path = run_dir / "RUN.fdf"
        if len(fc_files) != 1 or not fdf_path.is_file():
            raise PhononContractError(
                f"{run_dir}: expected exactly one *.FC and a RUN.fdf, found {len(fc_files)} FC files"
            )
        force_constants = force_constants_from_siesta_run(fc_files[0], fdf_path, **kwargs)
        return cls(
            force_constants=force_constants,
            apply_asr=apply_asr,
            geometry_signature=geometry_cell_species_sha256(fdf_path),
        )

    def modes(self, q_fractional: Sequence[float] = (0.0, 0.0, 0.0)) -> PhononModeSet:
        force_constants = self.force_constants
        if self.apply_asr:
            force_constants = force_constants.apply_asr()
        return modes_from_force_constants(
            force_constants,
            q_fractional,
            provider=self.backend,
            provider_version=self.provider_version,
            geometry_signature=self.geometry_signature,
        )


def read_vibra_vectors(vectors_path: Path) -> list[dict[str, Any]]:
    """Parse a VIBRA ``.vectors`` file into ``k``, frequencies (cm^-1) and modes.

    The format is the fixed block VIBRA writes: a ``k =`` line in 1/Bohr, then
    per branch ``Eigenvector``/``Frequency`` and the real and imaginary parts,
    one ``x y z`` row per atom.
    """
    # ponytail: 20 lines instead of sisl's vectorsSileSiesta, which loops
    # forever at EOF on a single-k file — and Gamma-only is exactly our case.
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    part: str | None = None
    for raw in Path(vectors_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("k") and "=" in line:
            current = {"k_1_bohr": [float(v) for v in line.split("=")[1].split()], "modes": []}
            blocks.append(current)
        elif line.startswith("Eigenvector"):
            current["modes"].append({"real": [], "imag": []})
        elif line.startswith("Frequency"):
            current["modes"][-1]["frequency_cm1"] = float(line.split("=")[1])
        elif line.startswith("Eigenmode"):
            part = "imag" if "imaginary" in line.lower() else "real"
        else:
            current["modes"][-1][part].append([float(value) for value in line.split()])
    if not blocks:
        raise PhononContractError(f"{vectors_path}: no 'k =' block found")
    return blocks


@dataclass(frozen=True)
class VibraPhononProvider:
    """Second backend: VIBRA's own output, converted at the adapter boundary.

    VIBRA builds ``D(q)`` with the phase of ``r_j - r_i``, i.e. the atomic phase
    convention, and prints cm^-1; both are declared here and removed by
    :func:`to_canonical_conventions`, never silently. It reports no ASR residual
    of its own, so the policy it is given must come from the FC run it used.

    ``source_conventions`` is the default *this repository asserts* for the
    installed VIBRA, in the same VERIFIED_BINARY_ONLY spirit as C01: it is a
    declaration to be checked against the release, which is why it is a field a
    run can override rather than a constant folded into the parser.
    """

    blocks: list[dict[str, Any]]
    cell_ang: np.ndarray
    positions_ang: np.ndarray
    masses_amu: np.ndarray
    asr_policy: str
    asr_residual_ev_ang2: float
    asr_residual_raw_ev_ang2: float
    source: dict[str, Any]
    geometry_signature: str = ""
    source_conventions: PhononConventions = PhononConventions(
        fourier_sign=1, atomic_phase=ATOMIC_PHASE_ATOM
    )

    backend = VIBRA_BACKEND
    provider_version = PROVIDER_VERSION

    @classmethod
    def from_vectors_file(
        cls,
        vectors_path: Path,
        force_constants: ForceConstants,
        **kwargs: Any,
    ) -> "VibraPhononProvider":
        """Pair a ``.vectors`` file with the FC run that produced it (for the ASR record)."""
        vectors_path = Path(vectors_path)
        return cls(
            blocks=read_vibra_vectors(vectors_path),
            cell_ang=force_constants.cell_ang,
            positions_ang=force_constants.positions_ang,
            masses_amu=force_constants.masses_amu,
            asr_policy=force_constants.asr_policy,
            asr_residual_ev_ang2=force_constants.asr_residual_ev_ang2,
            asr_residual_raw_ev_ang2=float(force_constants.asr_residual_raw_ev_ang2),
            source={
                "backend": VIBRA_BACKEND,
                "vectors_file": str(vectors_path),
                "vectors_sha256": file_sha256(vectors_path),
                **dict(force_constants.provenance or {}),
            },
            **kwargs,
        )

    def _q_fractional(self, block: dict[str, Any]) -> np.ndarray:
        # VIBRA prints k in 1/Bohr; q_frac = k . a / (2 pi).
        from fdf_materialization import BOHR_TO_ANG

        k_cartesian = np.asarray(block["k_1_bohr"], dtype=np.float64) / BOHR_TO_ANG
        return (self.cell_ang @ k_cartesian) / (2.0 * np.pi)

    def modes(self, q_fractional: Sequence[float] = (0.0, 0.0, 0.0)) -> PhononModeSet:
        target = np.asarray(q_fractional, dtype=np.float64).reshape(3)
        for block in self.blocks:
            found = self._q_fractional(block)
            if not np.allclose(found, target, atol=1e-6):
                continue
            frequencies = np.array([mode["frequency_cm1"] for mode in block["modes"]]) * CM1_TO_EV
            eigenvectors = np.array(
                [np.array(mode["real"]) + 1j * np.array(mode["imag"]) for mode in block["modes"]]
            )
            mode_set = PhononModeSet(
                q_fractional=found,
                frequencies_ev=frequencies,
                eigenvectors=eigenvectors,
                masses_amu=self.masses_amu,
                cell_ang=self.cell_ang,
                positions_ang=self.positions_ang,
                conventions=self.source_conventions,
                provider=self.backend,
                provider_version=self.provider_version,
                force_source=dict(self.source),
                primitive_mapping={
                    "atom_count": int(self.positions_ang.shape[0]),
                    "positions_ang": self.positions_ang.tolist(),
                    "cell_ang": self.cell_ang.tolist(),
                },
                fc_range={"source": "vibra", "cell_images": "declared_by_the_fc_run"},
                asr_policy=self.asr_policy,
                asr_residual_ev_ang2=self.asr_residual_ev_ang2,
                asr_residual_raw_ev_ang2=self.asr_residual_raw_ev_ang2,
                geometry_signature=self.geometry_signature,
            )
            return to_canonical_conventions(mode_set)
        available = [self._q_fractional(block).tolist() for block in self.blocks]
        raise PhononContractError(f"q={target.tolist()} is not in the file; it has {available}")
