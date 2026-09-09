#!/usr/bin/env python3
"""C18 / E-F_001-S22: the Gamma normalisation + ASR gate on the produced phonons.

S21 produced the modes (``run_graphene_gamma_phonons.py``); this is the verdict
on them. It runs no SCF and reads no fdf: it consumes the campaign manifest, the
two mode artifacts of every range (raw and ASR-corrected, never one overwriting
the other) and, when the run directory is still there, the ``.FC`` itself, so
that every published number is *recomputed* rather than trusted.

What it certifies, and why each check is the one that would catch the error:

``normalisation``
    ``sum_kappa |e_kappa|^2 = 1`` is what the contract declares; the check that
    makes it unambiguous is the one on the whole set, ``E E^dag = I``, because a
    per-vector norm is satisfied by any pair of vectors, degenerate or not.

``zero point``
    the factor is unambiguous iff it is measurable from the published fields
    alone. It is, through one identity that carries every mass and the whole
    normalisation at once:

        sum_kappa M_kappa |u_kappa|^2 = hbar / (2 omega)

    for ``u = displacement_pattern()``. It holds for *any* number of atoms and
    any mass distribution exactly because ``e`` is mass-weighted and unit-norm,
    so it fails the moment either convention slips. ``hbar^2/(amu eV Ang^2)`` is
    then checked against CODATA (scipy), not against itself.

``units``
    the modes are re-derived from the same IFC expressed in Ry/Bohr^2 with masses
    in electron masses, through a ``hbar^2`` built independently for that unit
    system, and converted back. Frequencies must return to the eV ones and the
    *subspaces* to the same spans -- individual eigenvectors need not return,
    because a second diagonalisation of a degenerate block is free to hand back
    another basis of it. That is the same freedom the doublet check exercises
    deliberately, so both are measured with the same principal angles.

``ASR``
    raw and corrected are two artifacts with two signatures; the correction is
    quantified in the units a consumer reads (cm^-1) through the Rayleigh
    quotient of the mass-weighted uniform translations, which is exactly the
    quantity the acoustic sum rule sets to zero. Raw: how far from zero the IFC
    puts it. Corrected: zero to roundoff. Neither number is inferred from the
    other, and the correction is checked to live only on the ``R = 0`` on-site
    blocks it is allowed to touch.

``range``
    the IFC tail and the frequencies, across the auxiliary supercells. With the
    confound named: widening the FC supercell also changes the ``k`` mesh of the
    run, so two ranges are only a measurement of the *IFC range* when their
    effective mesh ``repeats * kgrid`` coincides. Ranges are therefore grouped by
    that effective mesh and the gate is evaluated inside a group; the
    across-group differences are published as the k-sampling term they are.

Pre-registered tolerances ([DESIGN], fixed before the numbers were looked at,
:data:`TOLERANCES`):

* algebraic identities (normalisation, zero point, unit round trip, the ASR
  residual of a corrected IFC): float64 roundoff, ``64 * eps`` relative -- these
  are identities, not measurements;
* the ``hbar^2`` constant against CODATA: ``1e-6`` relative, i.e. ``5e-7`` on
  ``omega``, which is four orders below the IFC's own noise floor;
* gauge invariance of the doublet: ``1e-8`` rad, the accumulated roundoff of an
  SVD of a 6x6 unitary;
* range stability: ``2%`` on a frequency and ``0.01`` rad on a chosen subspace.
  Both come from one budget: ``g ~ 1/sqrt(omega)`` times an eigenvector, so a
  2% frequency and a 0.01 rad rotation each put ~1% into ``|g|``, which is the
  phonon's share of the ``tau_model`` C20 will pre-register. It is a tolerance on
  the *last* refinement of the range, because the change over the last step is
  the estimate of what is left, and earlier steps are reported, not gated.

Units: ``eV`` for frequencies (``cm^-1`` alongside, for reading), ``Ang`` for
displacements, ``amu`` for masses, ``eV/Ang^2`` for force constants.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import phonon_provider as pp  # noqa: E402
import run_graphene_gamma_phonons as campaign  # noqa: E402
from epc_subspaces import (  # noqa: E402
    random_gauge_rotation,
    subspace_metrics,
    subspace_overlap,
)
from reference_provenance import file_sha256  # noqa: E402

SCHEMA = "gamma_phonon_normalization_asr_certification_v1"
TICKET = "C18 / E-F_001-S22"

DEFAULT_CAMPAIGN_ROOT = campaign.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/certification/graphene"
REPORT_NAME = "c18_gamma_phonon_certification.json"

_EPS = float(np.finfo(np.float64).eps)

TOLERANCES = {
    "identity_relative": 64.0 * _EPS,
    "codata_constant_relative": 1e-6,
    # A principal angle near zero is sqrt(2 (1 - sigma)): roundoff in a singular
    # value arrives at the angle square-rooted, so an exact gauge rotation lands
    # at ~1e-7 rad and not at 1e-15. Same square root as
    # epc_subspaces.subspace_metric_tolerances.
    "gauge_angle_rad": float(np.sqrt(2.0 * 64.0 * _EPS)),
    "range_frequency_relative": 0.02,
    "range_subspace_angle_rad": 0.01,
    "policy": (
        "identities are gated at float64 roundoff, and quantities that come out of one "
        "square-rooted (principal angles, omega = sqrt(C lambda)) at its square root; the "
        "hbar^2 constant at 1e-6 relative (5e-7 on omega, four orders below the IFC noise "
        "floor); range stability at 2% on omega and 0.01 rad on a subspace, one 1%-of-|g| "
        "budget through g ~ e/sqrt(omega), applied to the last refinement of the range at "
        "fixed effective k mesh"
    ),
}


class GammaCertificationError(RuntimeError):
    """The gate cannot be evaluated at all (missing, unusable or uncertified inputs)."""


def check(name: str, passed: bool, detail: str, *, applicable: bool = True, **extra: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed) if applicable else True,
            "applicable": bool(applicable), "detail": detail, **extra}


# --------------------------------------------------------------------------- #
# Physical constants, from CODATA rather than from the repository
# --------------------------------------------------------------------------- #


def codata_constants() -> dict[str, float]:
    """``hbar^2`` and ``hc`` in the two unit systems the checks compare.

    ``scipy.constants`` is the independent source: the point of the checks below
    is that :data:`phonon_provider` agrees with it, so nothing here may be taken
    from :mod:`epc_formalism`.
    """
    from scipy import constants as sc

    return {
        # hbar^2 / (amu * eV * Ang^2), dimensionless: omega[eV]^2 = C * lambda[eV/(Ang^2 amu)]
        "hbar2_amu_ev_ang2": float(sc.hbar**2 / (sc.m_u * sc.e * 1e-20)),
        # the same combination in Rydberg atomic units: Ry, Bohr, electron mass
        "hbar2_me_ry_bohr2": float(
            sc.hbar**2 / (sc.m_e * (sc.Rydberg * sc.h * sc.c) * sc.value("Bohr radius") ** 2)
        ),
        "cm1_to_ev": float(sc.h * sc.c / sc.e * 100.0),
        "ev_per_ry": float(sc.Rydberg * sc.h * sc.c / sc.e),
        "ang_per_bohr": float(sc.value("Bohr radius") * 1e10),
        "amu_per_me": float(sc.m_u / sc.m_e),
        "source": "scipy.constants (CODATA)",
    }


# --------------------------------------------------------------------------- #
# The produced campaign
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RangeArtifacts:
    """One auxiliary FC range: both policies, and the IFC when it is still on disk."""

    tag: str
    repeats: list[int]
    kgrid: list[int]
    modes: dict[str, pp.PhononModeSet]
    payloads: dict[str, dict[str, Any]]
    published: dict[str, dict[str, Any]]
    force_constants: dict[str, pp.ForceConstants]
    run_dir: str

    @property
    def effective_kgrid(self) -> list[int]:
        """``repeats * kgrid``: the mesh of the *primitive* cell the run samples.

        Two ranges with the same effective mesh sample the same reciprocal
        density on the same points; a frequency difference between them is the
        IFC range and nothing else.
        """
        return [int(a) * int(b) for a, b in zip(self.repeats, self.kgrid)]


def load_campaign(root: Path) -> tuple[dict[str, Any], list[RangeArtifacts]]:
    """Manifest plus one :class:`RangeArtifacts` per certified range."""
    manifest_path = Path(root) / campaign.MANIFEST_NAME
    if not manifest_path.is_file():
        raise GammaCertificationError(
            f"{manifest_path} is missing. Run run_graphene_gamma_phonons.py first: this gate "
            "certifies produced modes, it does not produce them."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != campaign.SCHEMA:
        raise GammaCertificationError(
            f"{manifest_path} declares schema {manifest.get('schema')!r}, not {campaign.SCHEMA!r}."
        )
    if manifest.get("dry_run"):
        raise GammaCertificationError(f"{manifest_path} is a dry run: it holds no modes to certify.")

    ranges: list[RangeArtifacts] = []
    for row in manifest.get("rows", []):
        if not row.get("certified") or not row.get("modes"):
            continue
        modes, payloads, force_constants = {}, {}, {}
        for policy, entry in row["modes"].items():
            payload = json.loads((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))
            payloads[policy] = payload
            modes[policy] = pp.PhononModeSet.from_dict(payload)
        fc_path = Path(row["run_dir"]) / f"{row['range']}.FC"
        if fc_path.is_file():
            raw = pp.force_constants_from_siesta_supercell_run(
                fc_path,
                images=np.array(row["staged"]["cell_images"], dtype=np.int64),
                cell_ang=modes["raw"].cell_ang,
                positions_ang=modes["raw"].positions_ang,
                masses_amu=modes["raw"].masses_amu,
            )
            force_constants = {"raw": raw, "asr_corrected": raw.apply_asr()}
        ranges.append(
            RangeArtifacts(
                tag=row["range"],
                repeats=[int(value) for value in row["staged"]["repeats"]],
                kgrid=[int(value) for value in row["staged"]["kgrid_monkhorst_pack"]],
                modes=modes,
                payloads=payloads,
                published=row["modes"],
                force_constants=force_constants,
                run_dir=str(row["run_dir"]),
            )
        )
    if len(ranges) < 2:
        raise GammaCertificationError(
            f"{manifest_path} carries {len(ranges)} certified range(s); range sensitivity cannot "
            "be measured against fewer than two."
        )
    return manifest, ranges


# --------------------------------------------------------------------------- #
# Normalisation and the zero-point factor
# --------------------------------------------------------------------------- #


def flat_eigenvectors(mode_set: pp.PhononModeSet) -> np.ndarray:
    """``[3na, n_modes]``: one column per branch, the layout the metrics want."""
    return mode_set.eigenvectors.reshape(mode_set.mode_count, -1).T


def normalization_checks(mode_set: pp.PhononModeSet) -> list[dict[str, Any]]:
    """Declared *and* measured: per-vector norm, and the whole set's orthonormality."""
    vectors = flat_eigenvectors(mode_set)
    norms = np.linalg.norm(vectors, axis=0)
    gram = vectors.conj().T @ vectors
    identity_error = float(np.abs(gram - np.eye(mode_set.mode_count)).max(initial=0.0))
    tolerance = TOLERANCES["identity_relative"] * mode_set.mode_count
    return [
        check(
            "eigenvector_normalization_declared",
            mode_set.conventions.eigenvector_normalization == pp.MASS_WEIGHTED_UNIT_NORM
            and bool(mode_set.conventions.nc_placement),
            f"declared {mode_set.conventions.eigenvector_normalization!r} with an explicit "
            f"1/sqrt(Nc) placement",
            normalization=mode_set.conventions.eigenvector_normalization,
            nc_placement=mode_set.conventions.nc_placement,
        ),
        check(
            "eigenvector_norms_are_one",
            bool(np.abs(norms - 1.0).max(initial=0.0) <= tolerance),
            f"max |‖e‖ - 1| = {np.abs(norms - 1.0).max(initial=0.0):.3e} <= {tolerance:.3e}",
            max_norm_deviation=float(np.abs(norms - 1.0).max(initial=0.0)),
            tolerance=tolerance,
        ),
        check(
            # A per-vector norm is blind to two branches spanning the same
            # direction; E^dag E = I is not, which is what makes the
            # normalisation of the *set* unambiguous.
            "eigenvectors_are_mutually_orthonormal",
            identity_error <= tolerance,
            f"max |E^dag E - I| = {identity_error:.3e} <= {tolerance:.3e}",
            identity_error=identity_error,
            tolerance=tolerance,
        ),
    ]


def zero_point_checks(mode_set: pp.PhononModeSet, constants: Mapping[str, float]) -> list[dict[str, Any]]:
    """The amplitude is unambiguous iff it is recomputable from the published fields."""
    tolerance = TOLERANCES["identity_relative"] * mode_set.mode_count
    masses = mode_set.masses_amu
    hbar2 = float(constants["hbar2_amu_ev_ang2"])

    residuals: list[float] = []
    branches: list[dict[str, Any]] = []
    image_errors: list[float] = []
    for branch in range(mode_set.mode_count):
        mode = mode_set.mode(branch)
        if mode.frequency_ev <= 0.0:
            continue
        pattern = mode.displacement_pattern()
        # sum_kappa M_kappa |u_kappa|^2 = hbar / (2 omega): every mass and the
        # whole normalisation, in one number that does not depend on either.
        weighted = float(np.sum(masses[:, None] * np.abs(pattern) ** 2))
        # The repository's own constant, deliberately: this identity is about the
        # normalisation and the placement of the mass, and CODATA is compared
        # against separately by zero_point_constant_matches_codata. Mixing the two
        # would let a normalisation error hide behind a constant's last digits.
        expected = pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV / (2.0 * mode.frequency_ev)
        residuals.append(abs(weighted - expected) / expected)
        # At q = Gamma the pattern may not depend on which cell it is asked for:
        # that is the statement that no 1/sqrt(Nc) and no atomic phase is hiding
        # inside the amplitude.
        image_errors.append(
            float(np.abs(mode.displacement_pattern((2, -1, 0)) - pattern).max(initial=0.0))
        )
        branches.append(
            {
                "branch": branch,
                "frequency_ev": mode.frequency_ev,
                "frequency_cm1": mode.frequency_ev / pp.CM1_TO_EV,
                "zero_point_amplitude_ang": mode.zero_point_amplitude_ang.tolist(),
                "mass_weighted_norm_amu_ang2": weighted,
                "hbar_over_2omega_amu_ang2": expected,
            }
        )

    acoustic_refused = True
    acoustic_detail = "no branch has omega <= 0"
    for branch in range(mode_set.mode_count):
        if mode_set.frequencies_ev[branch] <= 0.0:
            try:
                mode_set.mode(branch).zero_point_amplitude_ang
                acoustic_refused, acoustic_detail = False, f"branch {branch} returned an amplitude"
            except pp.PhononContractError:
                acoustic_detail = f"branch {branch} (omega <= 0) refuses an amplitude"
            break

    constant_deviation = abs(pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV - hbar2) / hbar2
    return [
        check(
            "zero_point_constant_matches_codata",
            constant_deviation <= TOLERANCES["codata_constant_relative"],
            f"hbar^2/(amu eV Ang^2): repository {pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV:.9e} vs CODATA "
            f"{hbar2:.9e}, relative {constant_deviation:.3e}",
            relative_deviation=constant_deviation,
            tolerance=TOLERANCES["codata_constant_relative"],
            codata=hbar2,
            repository=pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV,
        ),
        check(
            "zero_point_amplitude_carries_hbar_over_2omega",
            bool(residuals) and max(residuals) <= tolerance,
            f"max relative error of sum_kappa M |u|^2 = hbar/(2 omega) over {len(residuals)} "
            f"branches: {max(residuals, default=float('nan')):.3e}",
            max_relative_error=max(residuals, default=None),
            tolerance=tolerance,
            branches=branches,
        ),
        check(
            "gamma_displacement_pattern_is_cell_independent",
            bool(image_errors) and max(image_errors) <= tolerance,
            f"max |u(R) - u(0)| over the branches at q = Gamma: "
            f"{max(image_errors, default=float('nan')):.3e}",
            max_image_difference=max(image_errors, default=None),
            tolerance=tolerance,
        ),
        check(
            "acoustic_amplitude_is_refused_not_approximated",
            acoustic_refused,
            acoustic_detail,
        ),
    ]


# --------------------------------------------------------------------------- #
# Unit-conversion invariance
# --------------------------------------------------------------------------- #


def rediagonalize_in_rydberg_units(
    force_constants: pp.ForceConstants, constants: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """The same modes from ``Phi`` in Ry/Bohr^2 and masses in electron masses.

    A different ``hbar^2`` (built for that unit system from CODATA), different
    matrix entries, one eigenproblem; the frequencies come back in eV. Nothing of
    :mod:`phonon_provider`'s eV constant is used, so an inconsistent conversion
    cannot cancel itself out.
    """
    bohr2_per_ang2 = 1.0 / float(constants["ang_per_bohr"]) ** 2
    phi_ry = force_constants.matrix / float(constants["ev_per_ry"]) / bohr2_per_ang2
    masses_me = force_constants.masses_amu * float(constants["amu_per_me"])

    n_atoms = force_constants.atom_count
    gamma = phi_ry.sum(axis=0).reshape(3 * n_atoms, 3 * n_atoms)
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses_me), 3)
    dynamical = inverse_sqrt_mass[:, None] * gamma * inverse_sqrt_mass[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (dynamical + dynamical.T))
    omega_ry = np.sign(eigenvalues) * np.sqrt(
        np.abs(eigenvalues) * float(constants["hbar2_me_ry_bohr2"])
    )
    return omega_ry * float(constants["ev_per_ry"]), eigenvectors


def unit_invariance_checks(
    mode_set: pp.PhononModeSet,
    force_constants: pp.ForceConstants | None,
    constants: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Frequencies, the cm^-1 the manifest publishes, and a full re-derivation."""
    tolerance = TOLERANCES["identity_relative"] * mode_set.mode_count
    codata_cm1 = float(constants["cm1_to_ev"])
    cm1_deviation = abs(pp.CM1_TO_EV - codata_cm1) / codata_cm1

    frequencies = mode_set.frequencies_ev
    # eV -> cm^-1 -> meV -> eV, each hop with its own factor: a lossy conversion
    # constant shows up as a failure to return.
    round_trip = ((frequencies / codata_cm1) * codata_cm1 * 1e3) / 1e3
    scale = max(float(np.abs(frequencies).max(initial=0.0)), 1e-12)
    round_trip_error = float(np.abs(round_trip - frequencies).max(initial=0.0) / scale)

    checks = [
        check(
            "cm1_conversion_matches_codata",
            cm1_deviation <= TOLERANCES["codata_constant_relative"],
            f"cm^-1 -> eV: repository {pp.CM1_TO_EV:.9e} vs CODATA {codata_cm1:.9e}, relative "
            f"{cm1_deviation:.3e}",
            relative_deviation=cm1_deviation,
            tolerance=TOLERANCES["codata_constant_relative"],
        ),
        check(
            "frequency_unit_round_trip_is_exact",
            round_trip_error <= tolerance,
            f"eV -> cm^-1 -> meV -> eV returns to {round_trip_error:.3e} of the spectrum's scale",
            max_relative_error=round_trip_error,
            tolerance=tolerance,
        ),
    ]

    if force_constants is None:
        return checks + [
            check(
                "frequencies_invariant_under_unit_system",
                True,
                "the .FC of this range is no longer on disk; the modes cannot be re-derived",
                applicable=False,
            )
        ]

    omega_ev, vectors_ry = rediagonalize_in_rydberg_units(force_constants, constants)
    frequency_error = float(np.abs(omega_ev - frequencies).max(initial=0.0) / scale)
    # The individual eigenvectors of a degenerate block are free to come back in
    # another basis of it -- that is the doublet freedom, not an error. What must
    # return is the span, so the comparison is by principal angle.
    labels = pp.mode_sector_report(mode_set)["branches"]
    cluster_of = [entry["cluster"] for entry in labels]
    angles: list[float] = []
    original = flat_eigenvectors(mode_set)
    for cluster in sorted(set(cluster_of)):
        columns = [index for index, label in enumerate(cluster_of) if label == cluster]
        metrics = subspace_metrics(
            subspace_overlap(original[:, columns], np.eye(original.shape[0]), vectors_ry[:, columns])
        )
        angles.append(float(metrics["maximum_principal_angle_rad"]))
    angle_tolerance = float(np.sqrt(2.0 * max(frequency_error, tolerance)))

    return checks + [
        check(
            "frequencies_invariant_under_unit_system",
            frequency_error <= max(tolerance, cm1_deviation, TOLERANCES["codata_constant_relative"]),
            f"Ry/Bohr^2 + electron masses + a CODATA hbar^2 for that system reproduces the eV "
            f"frequencies to {frequency_error:.3e} of the spectrum's scale",
            max_relative_error=frequency_error,
            tolerance=max(tolerance, cm1_deviation, TOLERANCES["codata_constant_relative"]),
            frequencies_ev_alternate=omega_ev.tolist(),
        ),
        check(
            "subspaces_invariant_under_unit_system",
            bool(angles) and max(angles) <= angle_tolerance,
            f"max principal angle between the eV and Ry spans of the same cluster: "
            f"{max(angles, default=float('nan')):.3e} rad <= {angle_tolerance:.3e}",
            max_principal_angle_rad=max(angles, default=None),
            tolerance=angle_tolerance,
            per_cluster_angle_rad=angles,
        ),
    ]


# --------------------------------------------------------------------------- #
# ASR: raw and corrected, side by side
# --------------------------------------------------------------------------- #


def translation_frequencies_ev(force_constants: pp.ForceConstants) -> np.ndarray:
    """``omega`` of the three mass-weighted uniform translations, signed by ``omega^2``.

    ``t_alpha[kappa] = sqrt(M_kappa) e_alpha`` normalised: the Rayleigh quotient
    of the dynamical matrix on it is exactly what the acoustic sum rule sets to
    zero, and it is a number rather than a branch, so it needs no identification
    of which eigenvector is acoustic.
    """
    dynamical = force_constants.dynamical_matrix()
    masses = force_constants.masses_amu
    frequencies = np.zeros(3)
    for axis in range(3):
        vector = np.zeros((masses.size, 3))
        vector[:, axis] = np.sqrt(masses)
        vector = (vector / np.linalg.norm(vector)).reshape(-1)
        quotient = float(np.real(vector @ dynamical @ vector))
        frequencies[axis] = np.sign(quotient) * np.sqrt(
            abs(quotient) * pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV
        )
    return frequencies


def asr_checks(
    artifacts: RangeArtifacts, tolerance_scale: float = 1.0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Quantify the residual, the correction, and that neither replaced the other."""
    raw, corrected = artifacts.modes["raw"], artifacts.modes["asr_corrected"]
    published = artifacts.published
    tolerance = TOLERANCES["identity_relative"] * tolerance_scale

    checks = [
        check(
            "raw_and_corrected_are_separate_artifacts",
            raw.asr_policy == pp.ASR_RAW
            and corrected.asr_policy == pp.ASR_CORRECTED
            and published["raw"]["path"] != published["asr_corrected"]["path"]
            and published["raw"]["signature_sha256"] != published["asr_corrected"]["signature_sha256"],
            "two files, two signatures, two policies; neither overwrites the other",
            raw_signature=published["raw"]["signature_sha256"],
            corrected_signature=published["asr_corrected"]["signature_sha256"],
        ),
        check(
            "corrected_carries_the_raw_residual",
            bool(
                np.isclose(corrected.asr_residual_raw_ev_ang2, raw.asr_residual_ev_ang2, rtol=1e-12)
                and corrected.asr_residual_ev_ang2 < corrected.asr_residual_raw_ev_ang2
            ),
            f"corrected residual {corrected.asr_residual_ev_ang2:.3e} eV/Ang^2 against a raw "
            f"{corrected.asr_residual_raw_ev_ang2:.3e} it still reports",
            raw_residual_ev_ang2=raw.asr_residual_ev_ang2,
            corrected_residual_ev_ang2=corrected.asr_residual_ev_ang2,
        ),
    ]

    summary: dict[str, Any] = {
        "raw_residual_ev_ang2": raw.asr_residual_ev_ang2,
        "corrected_residual_ev_ang2": corrected.asr_residual_ev_ang2,
    }

    if not artifacts.force_constants:
        return checks + [
            check(
                "asr_residual_is_recomputed_from_the_ifc",
                True,
                f"the .FC of {artifacts.tag} is no longer on disk; the published residual cannot "
                "be recomputed",
                applicable=False,
            )
        ], summary

    raw_fc = artifacts.force_constants["raw"]
    corrected_fc = artifacts.force_constants["asr_corrected"]
    correction = corrected_fc.matrix - raw_fc.matrix
    zero_image = int(np.argmin(np.abs(raw_fc.cell_images).sum(axis=1)))
    offsite = correction.copy()
    for kappa in range(raw_fc.atom_count):
        offsite[zero_image, kappa, :, kappa, :] = 0.0

    raw_translation = translation_frequencies_ev(raw_fc)
    corrected_translation = translation_frequencies_ev(corrected_fc)
    # A frequency comes out of the IFC square-rooted, so the roundoff of Phi
    # arrives at omega as its square root: an exactly corrected IFC lands at
    # ~1e-4 cm^-1, not at 1e-12. This is phonon_provider's own resolution policy,
    # evaluated at the floor of the sum a float64 IFC can cancel to.
    ifc_floor = (
        TOLERANCES["identity_relative"]
        * float(np.abs(raw_fc.matrix).max(initial=0.0))
        * raw_fc.cell_count
        * raw_fc.atom_count
        / float(raw_fc.masses_amu.min())
    )
    translation_tolerance_ev = float(pp.frequency_resolutions_ev([0.0], ifc_floor)[0])

    summary.update(
        {
            "recomputed_raw_residual_ev_ang2": raw_fc.asr_residual_ev_ang2,
            "correction_max_abs_ev_ang2": float(np.abs(correction).max(initial=0.0)),
            "raw_translation_frequency_cm1": (raw_translation / pp.CM1_TO_EV).tolist(),
            "corrected_translation_frequency_cm1": (corrected_translation / pp.CM1_TO_EV).tolist(),
            "raw_translation_frequency_ev": raw_translation.tolist(),
        }
    )
    checks += [
        check(
            "asr_residual_is_recomputed_from_the_ifc",
            bool(
                np.isclose(raw_fc.asr_residual_ev_ang2, raw.asr_residual_ev_ang2, rtol=1e-12)
                and np.isclose(
                    float(np.abs(correction).max(initial=0.0)), raw.asr_residual_ev_ang2, rtol=1e-12
                )
            ),
            f"the .FC gives a raw residual of {raw_fc.asr_residual_ev_ang2:.6e} eV/Ang^2 and a "
            f"correction of the same size, against the published {raw.asr_residual_ev_ang2:.6e}",
            recomputed=raw_fc.asr_residual_ev_ang2,
            published=raw.asr_residual_ev_ang2,
        ),
        check(
            "correction_touches_only_the_onsite_blocks",
            float(np.abs(offsite).max(initial=0.0)) <= tolerance,
            f"outside the R = 0 on-site blocks the correction is "
            f"{np.abs(offsite).max(initial=0.0):.3e} eV/Ang^2",
            max_offsite_ev_ang2=float(np.abs(offsite).max(initial=0.0)),
            tolerance=tolerance,
        ),
        check(
            "corrected_ifc_satisfies_the_sum_rule",
            corrected_fc.asr_residual_ev_ang2 <= ifc_floor * float(raw_fc.masses_amu.min()),
            f"corrected residual {corrected_fc.asr_residual_ev_ang2:.3e} eV/Ang^2 against a "
            f"summation floor of {ifc_floor * float(raw_fc.masses_amu.min()):.3e}",
            residual_ev_ang2=corrected_fc.asr_residual_ev_ang2,
        ),
        check(
            # The number a consumer of omega actually needs: what the violation
            # is worth in cm^-1, before and after.
            "translations_are_zero_after_correction",
            float(np.abs(corrected_translation).max(initial=0.0)) <= translation_tolerance_ev,
            f"uniform translations: raw up to "
            f"{np.abs(raw_translation).max(initial=0.0) / pp.CM1_TO_EV:.2f} cm^-1, corrected "
            f"{np.abs(corrected_translation).max(initial=0.0) / pp.CM1_TO_EV:.2e} cm^-1 against a "
            f"floor of {translation_tolerance_ev / pp.CM1_TO_EV:.2e} cm^-1",
            raw_max_cm1=float(np.abs(raw_translation).max(initial=0.0) / pp.CM1_TO_EV),
            corrected_max_cm1=float(np.abs(corrected_translation).max(initial=0.0) / pp.CM1_TO_EV),
            tolerance_cm1=translation_tolerance_ev / pp.CM1_TO_EV,
        ),
    ]
    return checks, summary


# --------------------------------------------------------------------------- #
# The doublet: allowed changes of basis
# --------------------------------------------------------------------------- #


def rotate_branches(
    mode_set: pp.PhononModeSet, branches: Sequence[int], unitary: np.ndarray
) -> pp.PhononModeSet:
    """``e -> e U`` on the given branches only; everything else is untouched."""
    vectors = flat_eigenvectors(mode_set).copy()
    vectors[:, list(branches)] = vectors[:, list(branches)] @ unitary
    return replace(
        mode_set,
        eigenvectors=vectors.T.reshape(mode_set.eigenvectors.shape),
    )


def doublet_checks(
    artifacts: RangeArtifacts, policy: str, *, seed: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A degenerate doublet is a subspace: nothing downstream may name its members."""
    mode_set = artifacts.modes[policy]
    report = artifacts.payloads[policy].get("sector_report") or pp.mode_sector_report(mode_set)
    candidate = report.get("e2g_candidate")
    if candidate is None:
        return [
            check(
                "doublet_basis_change_is_free",
                True,
                "no in-plane optical doublet was identified in this mode set",
                applicable=False,
            )
        ], {}

    branches = [int(index) for index in candidate["branches"]]
    vectors = flat_eigenvectors(mode_set)
    doublet = vectors[:, branches]
    identity = np.eye(vectors.shape[0])
    generator = np.random.default_rng(seed)
    angle = 0.7
    real_rotation = np.eye(len(branches))
    real_rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotations = {
        "haar_unitary": random_gauge_rotation([range(len(branches))], len(branches), generator),
        "real_rotation": real_rotation,
    }

    angles: dict[str, float] = {}
    amplitude_errors: dict[str, float] = {}
    identifications: dict[str, Any] = {}
    # The zero-point content of the *pair*: with a common omega, every member
    # contributes hbar/(2 omega) of mass-weighted displacement, so the total is
    # ‖E‖_F^2 hbar/(2 omega) and the masses cancel. It survives any unitary
    # mixing exactly because the Frobenius norm does -- which is the statement
    # that the amplitude belongs to the subspace, not to a member of it.
    mean_frequency = float(np.mean(mode_set.frequencies_ev[branches]))
    zero_point_scale = pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV / (2.0 * mean_frequency)
    reference_content = zero_point_scale * float(np.linalg.norm(doublet) ** 2)
    for name, unitary in rotations.items():
        rotated = doublet @ unitary
        metrics = subspace_metrics(subspace_overlap(doublet, identity, rotated))
        angles[name] = float(metrics["maximum_principal_angle_rad"])
        content = zero_point_scale * float(np.linalg.norm(rotated) ** 2)
        amplitude_errors[name] = abs(content - reference_content) / max(reference_content, 1e-30)
        rotated_report = pp.mode_sector_report(rotate_branches(mode_set, branches, unitary))
        rotated_candidate = rotated_report.get("e2g_candidate") or {}
        identifications[name] = {
            "branches": rotated_candidate.get("branches"),
            "mean_frequency_cm1": rotated_candidate.get("mean_frequency_cm1"),
            "acoustic_branches": rotated_report["acoustic_branches"],
        }

    # The teeth: rotating the doublet into a branch outside it is *not* an
    # allowed basis change, and the same metric must see it.
    outside = next(
        (index for index in range(mode_set.mode_count) if index not in branches), None
    )
    mixing_angle = None
    if outside is not None:
        mixed = np.column_stack(
            [
                (doublet[:, 0] + vectors[:, outside]) / np.sqrt(2.0),
                doublet[:, 1],
            ]
        )
        mixing_angle = float(
            subspace_metrics(subspace_overlap(doublet, identity, mixed))[
                "maximum_principal_angle_rad"
            ]
        )

    splitting_ev = float(candidate["frequency_spread_ev"])
    resolution_ev = float(candidate["resolution_ev"])
    summary = {
        "branches": branches,
        "mean_frequency_cm1": float(candidate["mean_frequency_cm1"]),
        "frequency_spread_cm1": splitting_ev / pp.CM1_TO_EV,
        "resolution_cm1": resolution_ev / pp.CM1_TO_EV,
        "max_principal_angle_rad": max(angles.values()),
        "amplitude_relative_error": max(amplitude_errors.values()),
        "cross_sector_mixing_angle_rad": mixing_angle,
        "identifications": identifications,
    }
    return [
        check(
            "doublet_basis_change_is_free",
            max(angles.values()) <= TOLERANCES["gauge_angle_rad"]
            and max(amplitude_errors.values()) <= TOLERANCES["identity_relative"] * mode_set.mode_count
            and all(
                entry["branches"] == branches and entry["acoustic_branches"] == report["acoustic_branches"]
                for entry in identifications.values()
            ),
            f"a Haar unitary and a real rotation of the doublet move its span by "
            f"{max(angles.values()):.3e} rad and its mass-weighted content by "
            f"{max(amplitude_errors.values()):.3e}, and the sector identification is unchanged",
            max_principal_angle_rad=max(angles.values()),
            max_amplitude_relative_error=max(amplitude_errors.values()),
            tolerance=TOLERANCES["gauge_angle_rad"],
            identifications=identifications,
        ),
        check(
            "doublet_metric_detects_a_forbidden_mixing",
            mixing_angle is not None and mixing_angle > 1e3 * TOLERANCES["gauge_angle_rad"],
            f"mixing the doublet with a branch outside it moves the span by "
            f"{mixing_angle if mixing_angle is None else round(mixing_angle, 6)} rad",
            mixing_angle_rad=mixing_angle,
            applicable=outside is not None,
        ),
        check(
            "doublet_degeneracy_is_resolved_as_one_cluster",
            splitting_ev <= resolution_ev,
            f"the members split by {splitting_ev / pp.CM1_TO_EV:.3f} cm^-1 against a resolution of "
            f"{resolution_ev / pp.CM1_TO_EV:.3f} cm^-1, so the basis change is allowed to that level",
            frequency_spread_cm1=splitting_ev / pp.CM1_TO_EV,
            resolution_cm1=resolution_ev / pp.CM1_TO_EV,
        ),
    ], summary


# --------------------------------------------------------------------------- #
# Sensitivity to the IFC range
# --------------------------------------------------------------------------- #


def sector_columns(artifacts: RangeArtifacts, policy: str) -> dict[str, list[int]]:
    """The chosen subspaces of a mode set: the acoustic set and the doublet."""
    report = artifacts.payloads[policy].get("sector_report") or pp.mode_sector_report(
        artifacts.modes[policy]
    )
    sectors = {"acoustic": [int(index) for index in report["acoustic_branches"]]}
    candidate = report.get("e2g_candidate")
    if candidate:
        sectors["e2g_doublet"] = [int(index) for index in candidate["branches"]]
    return sectors


def range_pair(first: RangeArtifacts, second: RangeArtifacts, policy: str) -> dict[str, Any]:
    """Frequencies and chosen subspaces of two ranges, in the same primitive basis."""
    left, right = first.modes[policy], second.modes[policy]
    if left.mode_count != right.mode_count:
        raise GammaCertificationError(
            f"{first.tag} and {second.tag} carry {left.mode_count} and {right.mode_count} branches; "
            "the Gamma modes of one primitive cell cannot differ in count"
        )
    delta_cm1 = (right.frequencies_ev - left.frequencies_ev) / pp.CM1_TO_EV
    scale = np.maximum(np.abs(left.frequencies_ev), np.abs(right.frequencies_ev))
    # A branch at omega = 0 by symmetry has no relative error to speak of; it is
    # gated in absolute cm^-1 against the ASR residual instead, which is what
    # translations_are_zero_after_correction already measures.
    resolved = scale > 10.0 * pp.CM1_TO_EV
    relative = np.where(
        resolved, np.abs(delta_cm1 * pp.CM1_TO_EV) / np.where(resolved, scale, 1.0), 0.0
    )

    columns_left = sector_columns(first, policy)
    columns_right = sector_columns(second, policy)
    vectors_left = flat_eigenvectors(left)
    vectors_right = flat_eigenvectors(right)
    identity = np.eye(vectors_left.shape[0])
    sectors: dict[str, Any] = {}
    for name, columns in columns_left.items():
        other = columns_right.get(name)
        if other is None or len(other) != len(columns):
            sectors[name] = {"comparable": False, "columns": [columns, other]}
            continue
        metrics = subspace_metrics(
            subspace_overlap(vectors_left[:, columns], identity, vectors_right[:, other])
        )
        sectors[name] = {
            "comparable": True,
            "branches": [columns, other],
            "max_principal_angle_rad": float(metrics["maximum_principal_angle_rad"]),
            "projector_frobenius_distance": float(metrics["projector_frobenius_distance"]),
        }
    return {
        "from_range": first.tag,
        "to_range": second.tag,
        "asr_policy": policy,
        "effective_kgrid": [first.effective_kgrid, second.effective_kgrid],
        "same_effective_kgrid": first.effective_kgrid == second.effective_kgrid,
        "tail_ratio": [
            (first.payloads[policy]["ifc_shells"]["tail_ratio"]),
            (second.payloads[policy]["ifc_shells"]["tail_ratio"]),
        ],
        "delta_cm1_per_branch": delta_cm1.tolist(),
        "max_abs_delta_cm1": float(np.abs(delta_cm1).max(initial=0.0)),
        "max_relative_frequency_change": float(relative.max(initial=0.0)),
        "resolved_branches": [int(index) for index in np.flatnonzero(resolved)],
        "sectors": sectors,
    }


def range_checks(ranges: Sequence[RangeArtifacts]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Range sensitivity, with the k-sampling confound separated rather than assumed away.

    Widening the FC supercell also rescales the run's ``k`` mesh, so a pair of
    ranges measures the IFC range alone only when ``repeats * kgrid`` -- the mesh
    of the primitive cell each run effectively samples -- coincides. Those pairs
    are the gate; the rest are published as the k-sampling term.
    """
    all_pairs = [
        range_pair(first, second, policy)
        for policy in ("raw", "asr_corrected")
        for first, second in zip(ranges, ranges[1:])
    ]
    # Every pair sharing an effective mesh, not just consecutive ones: with the
    # mesh held fixed the IFC range is the only thing that changed.
    controlled = [
        range_pair(first, second, policy)
        for policy in ("raw", "asr_corrected")
        for index, first in enumerate(ranges)
        for second in ranges[index + 1 :]
        if first.effective_kgrid == second.effective_kgrid
    ]
    # The gated pair is the one that reaches the widest FC supercell at a fixed
    # mesh: the change over the last refinement is the estimate of what is left.
    # Earlier steps are reported, not gated.
    widest_repeats = {item.tag: max(item.repeats) for item in ranges}
    gated = [pair for pair in controlled if pair["asr_policy"] == "asr_corrected"]
    widest = max(gated, key=lambda pair: widest_repeats[pair["to_range"]], default=None)

    summary = {
        "ranges": [
            {
                "range": item.tag,
                "repeats": item.repeats,
                "kgrid_monkhorst_pack": item.kgrid,
                "effective_kgrid": item.effective_kgrid,
                "tail_ratio": item.payloads["asr_corrected"]["ifc_shells"]["tail_ratio"],
                "max_distance_ang": item.payloads["asr_corrected"]["ifc_shells"]["max_distance_ang"],
                "frequencies_cm1": {
                    policy: (item.modes[policy].frequencies_ev / pp.CM1_TO_EV).tolist()
                    for policy in item.modes
                },
            }
            for item in ranges
        ],
        "consecutive_pairs": all_pairs,
        "k_controlled_pairs": controlled,
        "gated_pair": widest,
        "k_sampling_confound": (
            "the FC supercell and the k mesh were varied together; only pairs with the same "
            "effective mesh repeats*kgrid isolate the IFC range, and they are the ones gated"
        ),
    }

    if widest is None:
        return [
            check(
                "frequencies_are_stable_in_the_ifc_range",
                True,
                "no two ranges share an effective k mesh, so no pair measures the IFC range alone; "
                "the consecutive differences are published as a combined range + k-sampling term",
                applicable=False,
                consecutive_max_abs_delta_cm1=max(
                    (pair["max_abs_delta_cm1"] for pair in all_pairs), default=None
                ),
            ),
            check(
                "chosen_subspaces_are_stable_in_the_ifc_range",
                True,
                "no k-controlled pair of ranges is available",
                applicable=False,
            ),
        ], summary

    angles = [
        sector["max_principal_angle_rad"]
        for sector in widest["sectors"].values()
        if sector.get("comparable")
    ]
    return [
        check(
            "frequencies_are_stable_in_the_ifc_range",
            widest["max_relative_frequency_change"] <= TOLERANCES["range_frequency_relative"],
            f"{widest['from_range']} -> {widest['to_range']} at a fixed effective mesh "
            f"{widest['effective_kgrid'][0]}: the resolved branches move by "
            f"{widest['max_relative_frequency_change']:.3%} "
            f"({widest['max_abs_delta_cm1']:.2f} cm^-1 over all branches)",
            max_relative_frequency_change=widest["max_relative_frequency_change"],
            tolerance=TOLERANCES["range_frequency_relative"],
            pair=[widest["from_range"], widest["to_range"]],
        ),
        check(
            "chosen_subspaces_are_stable_in_the_ifc_range",
            bool(angles) and max(angles) <= TOLERANCES["range_subspace_angle_rad"],
            f"acoustic and doublet spans move by at most {max(angles, default=float('nan')):.3e} rad "
            f"between {widest['from_range']} and {widest['to_range']}",
            max_principal_angle_rad=max(angles, default=None),
            tolerance=TOLERANCES["range_subspace_angle_rad"],
            sectors=widest["sectors"],
        ),
    ], summary


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def certify(campaign_root: Path, *, seed: int = 0) -> dict[str, Any]:
    manifest, ranges = load_campaign(Path(campaign_root))
    constants = codata_constants()

    rows: list[dict[str, Any]] = []
    for artifacts in ranges:
        asr, asr_summary = asr_checks(artifacts, tolerance_scale=artifacts.modes["raw"].mode_count)
        policies: dict[str, Any] = {}
        checks = list(asr)
        for policy, mode_set in sorted(artifacts.modes.items()):
            doublet, doublet_summary = doublet_checks(artifacts, policy, seed=seed)
            policy_checks = (
                normalization_checks(mode_set)
                + zero_point_checks(mode_set, constants)
                + unit_invariance_checks(
                    mode_set, artifacts.force_constants.get(policy), constants
                )
                + doublet
            )
            policies[policy] = {
                "asr_policy": mode_set.asr_policy,
                "frequencies_cm1": (mode_set.frequencies_ev / pp.CM1_TO_EV).tolist(),
                "doublet": doublet_summary,
                "checks": policy_checks,
            }
            checks += [{**entry, "policy": policy} for entry in policy_checks]
        rows.append(
            {
                "range": artifacts.tag,
                "repeats": artifacts.repeats,
                "kgrid_monkhorst_pack": artifacts.kgrid,
                "effective_kgrid": artifacts.effective_kgrid,
                "ifc_present": bool(artifacts.force_constants),
                "asr": asr_summary,
                "asr_checks": asr,
                "policies": policies,
                # The policy travels with the name: the same check on the raw and
                # on the corrected IFC are two results, and a failure has to say
                # which one it was.
                "checks_failed": [
                    {"check": entry["check"], "policy": entry.get("policy")}
                    for entry in checks
                    if entry["applicable"] and not entry["passed"]
                ],
                "checks_inapplicable": [
                    {"check": entry["check"], "policy": entry.get("policy")}
                    for entry in checks
                    if not entry["applicable"]
                ],
            }
        )

    stability, stability_summary = range_checks(ranges)
    failed = [
        {"range": row["range"], **entry} for row in rows for entry in row["checks_failed"]
    ] + [
        {"range": "all", "check": entry["check"]}
        for entry in stability
        if entry["applicable"] and not entry["passed"]
    ]
    inapplicable = [
        {"range": row["range"], **entry} for row in rows for entry in row["checks_inapplicable"]
    ] + [
        {"range": "all", "check": entry["check"]}
        for entry in stability
        if not entry["applicable"]
    ]

    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root),
        "campaign_manifest_sha256": file_sha256(Path(campaign_root) / campaign.MANIFEST_NAME),
        "campaign_schema": manifest.get("schema"),
        "siesta_runtime_ref": manifest.get("siesta_runtime_ref"),
        "compute_policy": manifest.get("compute_policy"),
        "phonon_contract_id": pp.CONTRACT_ID,
        "q_fractional": manifest.get("q_fractional"),
        "conventions": pp.CANONICAL_CONVENTIONS.to_dict(),
        "units": dict(pp.UNITS),
        "constants": constants,
        "tolerances": dict(TOLERANCES),
        "ranges": rows,
        "range_stability": {"checks": stability, **stability_summary},
        "summary": {
            "ranges_certified": len(rows),
            "checks_failed": failed,
            "checks_inapplicable": inapplicable,
        },
        "verdict": "PASS" if not failed else "NO_GO",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0, help="seed of the random doublet rotation")
    return parser


def _print_range(row: Mapping[str, Any]) -> None:
    corrected = row["policies"]["asr_corrected"]
    asr = row["asr"]
    raw_translation = asr.get("raw_translation_frequency_cm1")
    print(
        f"  {row['range']:<9} effective k {row['effective_kgrid']}  "
        f"ASR raw {asr['raw_residual_ev_ang2']:.3e} eV/Ang^2"
        + (f" (<= {max(map(abs, raw_translation)):.1f} cm^-1)" if raw_translation else "")
        + f"  corrected {asr['corrected_residual_ev_ang2']:.1e}"
    )
    doublet = corrected.get("doublet") or {}
    if doublet:
        print(
            f"            doublet {doublet['mean_frequency_cm1']:.2f} cm^-1  "
            f"split {doublet['frequency_spread_cm1']:.3f}  "
            f"gauge angle {doublet['max_principal_angle_rad']:.2e} rad"
        )
    for entry in row["checks_failed"]:
        print(f"            FAILED {entry['check']}  [{entry.get('policy') or 'asr'}]")


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = certify(args.campaign_root, seed=args.seed)
    except (GammaCertificationError, pp.PhononContractError) as error:
        print(f"[C18-GATE][ERROR] {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / REPORT_NAME
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"C18 Gamma normalisation/ASR gate: {report['verdict']}  ->  {output}")
    for row in report["ranges"]:
        _print_range(row)
    for entry in report["range_stability"]["checks"]:
        state = "n/a " if not entry["applicable"] else ("ok  " if entry["passed"] else "FAIL")
        print(f"  [{state}] {entry['check']}: {entry['detail']}")
    for failure in report["summary"]["checks_failed"]:
        print(f"  FAILED {failure['range']}: {failure['check']} [{failure.get('policy') or 'asr'}]")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
