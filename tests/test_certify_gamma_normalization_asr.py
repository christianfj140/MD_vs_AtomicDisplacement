"""C18 / E-F_001-S22: the Gamma normalisation + ASR gate.

The gate's job is to fail when something is wrong, so every check here is
exercised twice: once on an object that is right, and once on one that is
deliberately broken in the single way that check exists to catch. A check that
only ever passes certifies nothing.

The toy is a two-atom spring model, ``Phi = [[K, -K], [-K, K]]``, whose acoustic
sum rule holds by construction and whose spectrum is known in closed form: three
zeros and ``omega^2 = C * 2 eig(K) / M``. Choosing ``K = diag(kx, kx, kz)`` with
``kx > kz`` puts an exact in-plane doublet at the top, which is the E2g the gate
looks for -- with no SIESTA and no fitting.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import certify_gamma_normalization_asr as gate  # noqa: E402
import phonon_provider as pp  # noqa: E402
import run_graphene_gamma_phonons as producer  # noqa: E402

CELL = np.array([[2.2005, -1.2704592674, 0.0], [2.2005, 1.2704592674, 0.0], [0.0, 0.0, 29.34]])
POSITIONS = np.array([[1.467, 0.0, 0.0], [2.934, 0.0, 0.0]])
CARBON = np.array([12.0107, 12.0107])
REPORT = gate.DEFAULT_OUTPUT_DIR / gate.REPORT_NAME


# --------------------------------------------------------------------------- #
# The toy
# --------------------------------------------------------------------------- #


def spring_force_constants(
    *,
    kx: float = 30.0,
    kz: float = 8.0,
    violation: np.ndarray | float = 0.25,
    masses: np.ndarray = CARBON,
) -> pp.ForceConstants:
    """``Phi = [[K, -K], [-K, K]]`` plus an on-site ASR violation.

    The violation is not decoration: the resolution at which
    :func:`phonon_provider.mode_sector_report` groups branches is derived from the
    *raw* translational residual, so an IFC that satisfies the sum rule to the
    last bit resolves an exact doublet into two singletons. A real IFC never
    does, and neither does this one. :func:`ForceConstants.apply_asr` removes it
    again and leaves the closed-form spectrum.
    """
    stiffness = np.diag([kx, kx, kz])
    matrix = np.zeros((1, 2, 3, 2, 3))
    for first in range(2):
        for second in range(2):
            matrix[0, first, :, second, :] = stiffness if first == second else -stiffness
    matrix[0, 0, :, 0, :] += violation
    matrix[0, 1, :, 1, :] += violation
    return pp.ForceConstants(
        matrix=matrix,
        cell_images=[[0, 0, 0]],
        cell_ang=CELL,
        positions_ang=POSITIONS,
        masses_amu=masses,
        provenance={"backend": "toy_spring_model"},
    )


def toy_modes(force_constants: pp.ForceConstants) -> pp.PhononModeSet:
    return pp.modes_from_force_constants(force_constants, provider="toy", geometry_signature="toy")


def toy_range(
    tag: str,
    *,
    repeats=(1, 1, 1),
    kgrid=(20, 20, 1),
    kx: float = 30.0,
    violation: float = 0.25,
) -> gate.RangeArtifacts:
    """A :class:`RangeArtifacts` with no files behind it: both policies, one IFC."""
    raw = spring_force_constants(kx=kx, violation=violation * np.eye(3))
    corrected = raw.apply_asr()
    modes = {"raw": toy_modes(raw), "asr_corrected": toy_modes(corrected)}
    payloads = {
        policy: {
            **mode_set.to_dict(),
            "ifc_shells": pp.ifc_shells(raw if policy == "raw" else corrected),
            "sector_report": pp.mode_sector_report(mode_set),
        }
        for policy, mode_set in modes.items()
    }
    published = {
        policy: {"path": f"{tag}/{policy}.json", "signature_sha256": f"{tag}-{policy}"}
        for policy in modes
    }
    return gate.RangeArtifacts(
        tag=tag,
        repeats=[int(value) for value in repeats],
        kgrid=[int(value) for value in kgrid],
        modes=modes,
        payloads=payloads,
        published=published,
        force_constants={"raw": raw, "asr_corrected": corrected},
        run_dir=f"/nowhere/{tag}",
    )


def by_name(checks, name):
    return next(entry for entry in checks if entry["check"] == name)


@pytest.fixture(scope="module")
def constants():
    return gate.codata_constants()


# --------------------------------------------------------------------------- #
# The toy is the toy it claims to be
# --------------------------------------------------------------------------- #


def test_the_spring_model_has_the_spectrum_the_gate_is_tested_against():
    modes = toy_modes(spring_force_constants(kx=30.0, kz=8.0).apply_asr())
    expected = np.sqrt(
        pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV * 2.0 * np.array([8.0, 30.0, 30.0]) / CARBON[0]
    )
    # omega = sqrt(C lambda), so the roundoff of an exactly zero lambda arrives
    # at omega square-rooted: 1e-16 in the matrix is 1e-9 in eV, not 1e-16.
    assert modes.frequencies_ev[:3] == pytest.approx(0.0, abs=1e-8)
    assert modes.frequencies_ev[3:] == pytest.approx(expected, rel=1e-12)
    doublet = pp.mode_sector_report(modes)["e2g_candidate"]
    assert doublet["branches"] == [4, 5] and doublet["size"] == 2


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_normalization_is_measured_on_the_set_not_only_per_vector():
    modes = toy_modes(spring_force_constants())
    checks = gate.normalization_checks(modes)
    assert all(entry["passed"] for entry in checks)

    # Two branches spanning the same direction: every norm is still 1, so only
    # E^dag E = I can see it. This is why the set-level check exists.
    vectors = gate.flat_eigenvectors(modes)
    vectors[:, 5] = vectors[:, 4]
    collapsed = replace(modes, eigenvectors=vectors.T.reshape(modes.eigenvectors.shape))
    checks = gate.normalization_checks(collapsed)
    assert by_name(checks, "eigenvector_norms_are_one")["passed"]
    assert not by_name(checks, "eigenvectors_are_mutually_orthonormal")["passed"]


# --------------------------------------------------------------------------- #
# The zero-point factor
# --------------------------------------------------------------------------- #


def test_zero_point_identity_holds_with_unequal_masses(constants):
    """``sum_kappa M |u|^2 = hbar/(2 omega)`` is where the masses have to cancel."""
    modes = toy_modes(spring_force_constants(masses=np.array([12.0107, 1.008])))
    checks = gate.zero_point_checks(modes, constants)
    assert all(entry["passed"] for entry in checks), [
        entry for entry in checks if not entry["passed"]
    ]
    identity = by_name(checks, "zero_point_amplitude_carries_hbar_over_2omega")
    assert identity["max_relative_error"] < 1e-13
    # Every branch with omega > 0 is measured, including an acoustic one that
    # roundoff left barely positive: the identity is exact there too, and a
    # divergent amplitude does not weaken it.
    assert len(identity["branches"]) >= 3


def test_an_nc_factor_hidden_in_the_amplitude_is_caught(monkeypatch, constants):
    """The failure mode the identity exists for: a 1/sqrt(Nc) folded into ``u``."""
    original = pp.PhononMode.displacement_pattern
    monkeypatch.setattr(
        pp.PhononMode,
        "displacement_pattern",
        lambda self, cell_image=(0, 0, 0): original(self, cell_image) * pp.nc_factor(9),
    )
    checks = gate.zero_point_checks(toy_modes(spring_force_constants()), constants)
    identity = by_name(checks, "zero_point_amplitude_carries_hbar_over_2omega")
    assert not identity["passed"]
    assert identity["max_relative_error"] == pytest.approx(1.0 - 1.0 / 9.0, rel=1e-9)


def test_the_acoustic_amplitude_is_refused(constants):
    checks = gate.zero_point_checks(toy_modes(spring_force_constants()), constants)
    assert by_name(checks, "acoustic_amplitude_is_refused_not_approximated")["passed"]
    assert by_name(checks, "zero_point_constant_matches_codata")["relative_deviation"] < 1e-6


# --------------------------------------------------------------------------- #
# Unit conversion
# --------------------------------------------------------------------------- #


def test_frequencies_and_subspaces_survive_a_change_of_unit_system(constants):
    force_constants = spring_force_constants()
    checks = gate.unit_invariance_checks(toy_modes(force_constants), force_constants, constants)
    assert all(entry["passed"] for entry in checks)
    assert by_name(checks, "frequencies_invariant_under_unit_system")["max_relative_error"] < 1e-6


def test_an_inconsistent_constant_between_unit_systems_is_caught(constants):
    """The Ry path is only a check if a wrong ``hbar^2`` there cannot cancel out."""
    force_constants = spring_force_constants()
    corrupted = {**constants, "hbar2_me_ry_bohr2": constants["hbar2_me_ry_bohr2"] * 1.01}
    checks = gate.unit_invariance_checks(toy_modes(force_constants), force_constants, corrupted)
    assert not by_name(checks, "frequencies_invariant_under_unit_system")["passed"]
    # The spans are unaffected: a wrong constant rescales omega, it does not
    # rotate an eigenvector. The two checks measure different things.
    assert by_name(checks, "subspaces_invariant_under_unit_system")["passed"]


def test_the_unit_checks_report_themselves_inapplicable_without_the_ifc(constants):
    checks = gate.unit_invariance_checks(toy_modes(spring_force_constants()), None, constants)
    entry = by_name(checks, "frequencies_invariant_under_unit_system")
    assert not entry["applicable"] and "no longer on disk" in entry["detail"]


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #


def test_the_translation_rayleigh_quotient_is_the_violation_in_cm1():
    """What the sum rule sets to zero, in the units a consumer of omega reads."""
    violation = 0.25
    raw = spring_force_constants(violation=violation * np.eye(3))
    frequencies = gate.translation_frequencies_ev(raw)
    expected = np.sqrt(pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV * violation / CARBON[0])
    assert frequencies == pytest.approx(np.repeat(expected, 3), rel=1e-12)
    assert gate.translation_frequencies_ev(raw.apply_asr()) == pytest.approx(0.0, abs=1e-14)


def test_asr_checks_quantify_the_residual_and_where_the_correction_lands():
    artifacts = toy_range("toy", violation=0.25)
    checks, summary = gate.asr_checks(artifacts, tolerance_scale=6)
    assert all(entry["passed"] for entry in checks), [
        entry for entry in checks if not entry["passed"]
    ]
    assert summary["raw_residual_ev_ang2"] == pytest.approx(0.25)
    assert summary["recomputed_raw_residual_ev_ang2"] == pytest.approx(0.25)
    assert summary["correction_max_abs_ev_ang2"] == pytest.approx(0.25)
    assert summary["corrected_residual_ev_ang2"] < 1e-14
    # Raw and corrected are two measurements, not one overwritten by the other.
    assert max(map(abs, summary["raw_translation_frequency_cm1"])) > 50.0
    assert max(map(abs, summary["corrected_translation_frequency_cm1"])) < 1e-6


def test_a_corrected_artifact_masquerading_as_the_raw_one_is_caught():
    artifacts = toy_range("toy", violation=0.25)
    same_file = {policy: dict(artifacts.published["raw"]) for policy in artifacts.published}
    checks, _ = gate.asr_checks(replace(artifacts, published=same_file), tolerance_scale=6)
    assert not by_name(checks, "raw_and_corrected_are_separate_artifacts")["passed"]


def test_an_ifc_that_never_violated_the_sum_rule_reports_no_correction():
    artifacts = toy_range("clean", violation=0.0)
    checks, summary = gate.asr_checks(artifacts, tolerance_scale=6)
    assert summary["raw_residual_ev_ang2"] == pytest.approx(0.0, abs=1e-14)
    # The residual is already zero, so "corrected < raw" cannot hold: the gate
    # says so rather than passing an empty statement.
    assert not by_name(checks, "corrected_carries_the_raw_residual")["passed"]


# --------------------------------------------------------------------------- #
# The doublet
# --------------------------------------------------------------------------- #


def test_the_doublet_may_be_rotated_and_the_metric_still_has_teeth():
    artifacts = toy_range("toy")
    checks, summary = gate.doublet_checks(artifacts, "asr_corrected")
    assert all(entry["passed"] for entry in checks), [
        entry for entry in checks if not entry["passed"]
    ]
    assert summary["branches"] == [4, 5]
    assert summary["max_principal_angle_rad"] < gate.TOLERANCES["gauge_angle_rad"]
    # A rotation out of the doublet is not an allowed basis change, and the same
    # metric that reports ~0 for the allowed one must report a real angle here.
    assert summary["cross_sector_mixing_angle_rad"] > 0.1
    for identification in summary["identifications"].values():
        assert identification["branches"] == [4, 5]
        assert identification["acoustic_branches"] == [0, 1, 2]


def test_a_split_doublet_is_reported_as_split_not_silently_rotated():
    """``kx != ky``: there is no doublet to rotate, and the gate must not invent one."""
    force_constants = spring_force_constants()
    force_constants.matrix[0, :, 1, :, 1] *= 1.5  # break the in-plane degeneracy
    modes = toy_modes(force_constants)
    report = pp.mode_sector_report(modes)
    candidate = report["e2g_candidate"]
    assert candidate is None or candidate["frequency_spread_ev"] > candidate["resolution_ev"]


# --------------------------------------------------------------------------- #
# Range sensitivity, and the k-sampling confound
# --------------------------------------------------------------------------- #


def test_only_ranges_sharing_an_effective_k_mesh_gate_the_ifc_range():
    """The middle range samples a different mesh: its frequency shift is not the range."""
    ranges = [
        toy_range("sc1x1x1", repeats=(1, 1, 1), kgrid=(20, 20, 1), kx=30.0),
        toy_range("sc3x3x1", repeats=(3, 3, 1), kgrid=(7, 7, 1), kx=24.0),  # 21x21: other mesh
        toy_range("sc5x5x1", repeats=(5, 5, 1), kgrid=(4, 4, 1), kx=30.01),
    ]
    checks, summary = gate.range_checks(ranges)
    assert all(entry["passed"] for entry in checks)
    assert summary["gated_pair"]["from_range"] == "sc1x1x1"
    assert summary["gated_pair"]["to_range"] == "sc5x5x1"
    assert summary["gated_pair"]["same_effective_kgrid"]
    # The k-driven excursion is published, not gated away.
    consecutive = [pair for pair in summary["consecutive_pairs"] if pair["asr_policy"] == "asr_corrected"]
    assert max(pair["max_relative_frequency_change"] for pair in consecutive) > 0.1
    assert not any(pair["same_effective_kgrid"] for pair in consecutive)


def test_a_range_that_moves_the_frequencies_fails_the_gate():
    ranges = [
        toy_range("sc1x1x1", repeats=(1, 1, 1), kgrid=(20, 20, 1), kx=30.0),
        toy_range("sc5x5x1", repeats=(5, 5, 1), kgrid=(4, 4, 1), kx=36.0),  # +10% in omega
    ]
    checks, summary = gate.range_checks(ranges)
    entry = by_name(checks, "frequencies_are_stable_in_the_ifc_range")
    assert not entry["passed"]
    assert entry["max_relative_frequency_change"] > gate.TOLERANCES["range_frequency_relative"]
    assert summary["gated_pair"]["same_effective_kgrid"]


def test_without_a_k_controlled_pair_the_gate_abstains():
    ranges = [
        toy_range("sc1x1x1", repeats=(1, 1, 1), kgrid=(20, 20, 1)),
        toy_range("sc3x3x1", repeats=(3, 3, 1), kgrid=(7, 7, 1)),
    ]
    checks, summary = gate.range_checks(ranges)
    assert summary["gated_pair"] is None
    for entry in checks:
        assert not entry["applicable"]
        assert entry["passed"]  # an inapplicable check is never a failure
    assert summary["consecutive_pairs"]


def test_the_gate_refuses_a_campaign_it_cannot_judge(tmp_path):
    with pytest.raises(gate.GammaCertificationError, match="is missing"):
        gate.load_campaign(tmp_path)
    (tmp_path / producer.MANIFEST_NAME).write_text(
        json.dumps({"schema": producer.SCHEMA, "dry_run": True, "rows": []}), encoding="utf-8"
    )
    with pytest.raises(gate.GammaCertificationError, match="dry run"):
        gate.load_campaign(tmp_path)
    (tmp_path / producer.MANIFEST_NAME).write_text(
        json.dumps({"schema": producer.SCHEMA, "rows": []}), encoding="utf-8"
    )
    with pytest.raises(gate.GammaCertificationError, match="fewer than two"):
        gate.load_campaign(tmp_path)


# --------------------------------------------------------------------------- #
# The produced certification
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (producer.DEFAULT_OUTPUT_ROOT / producer.MANIFEST_NAME).is_file(),
    reason="the Gamma phonon campaign has not been run here",
)
def test_the_produced_campaign_passes_the_gate():
    report = gate.certify(producer.DEFAULT_OUTPUT_ROOT)
    assert report["schema"] == gate.SCHEMA
    assert report["q_fractional"] == [0.0, 0.0, 0.0]
    assert report["summary"]["ranges_certified"] >= 2
    assert report["verdict"] == "PASS", report["summary"]["checks_failed"]

    for row in report["ranges"]:
        assert set(row["policies"]) == {"raw", "asr_corrected"}
        assert row["asr"]["raw_residual_ev_ang2"] > row["asr"]["corrected_residual_ev_ang2"]
        assert row["ifc_present"]

    gated = report["range_stability"]["gated_pair"]
    assert gated is not None and gated["same_effective_kgrid"]
    assert gated["asr_policy"] == "asr_corrected"


@pytest.mark.skipif(not REPORT.is_file(), reason="the C18 gate has not been run here")
def test_the_written_report_is_the_one_the_gate_produces():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == gate.SCHEMA and report["ticket"] == gate.TICKET
    assert report["verdict"] == "PASS"
    assert report["tolerances"]["range_frequency_relative"] == gate.TOLERANCES["range_frequency_relative"]
    assert report["constants"]["source"].startswith("scipy.constants")
    assert report["compute_policy"]["effective_backend"] == "cpu"
