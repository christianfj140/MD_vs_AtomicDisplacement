"""C17 / E-F_001-S20: the phonon contract, checked on the real graphene FC run.

What has to hold before any ``g`` may be labelled in eV: the units chain from
eV/Ang^2 force constants to an eV frequency and an Ang zero-point amplitude, the
declared conventions (Fourier sign, atomic phase, mass-weighted normalisation),
the separation of raw from ASR-corrected IFC, and the impossibility of passing a
synthetic benchmark displacement off as a phonon.

The physical anchors are the ones the roadmap cites for graphene: three acoustic
modes at Gamma that are exactly zero *after* ASR (and measurably non-zero
before), and an E2g doublet near 1580 cm^-1.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from artifact_signature import (  # noqa: E402
    PHYSICAL_PHONON,
    SYNTHETIC_TEST_DISPLACEMENT,
    EpcDagError,
    epc_artifact_node,
)
from epc_formalism import zero_point_amplitude_ang  # noqa: E402
from phonon_provider import (  # noqa: E402
    ASR_CORRECTED,
    ASR_RAW,
    ATOMIC_PHASE_ATOM,
    ATOMIC_PHASE_CELL,
    CANONICAL_CONVENTIONS,
    CM1_TO_EV,
    CONTRACT_ID,
    MASS_WEIGHTED_UNIT_NORM,
    UNITS,
    ForceConstants,
    PhononContractError,
    PhononConventions,
    PhononModeSet,
    SiestaFcPhononProvider,
    SyntheticTestDisplacement,
    VibraPhononProvider,
    force_constants_from_siesta_run,
    modes_from_force_constants,
    nc_factor,
    read_siesta_fc,
    require_physical_phonon,
    to_canonical_conventions,
)

FC_RUN = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene/runs/fc_dhs_d0p01"
CARBON_MASS_AMU = 12.0107

pytestmark = pytest.mark.skipif(not FC_RUN.is_dir(), reason="graphene FC run not archived here")


@pytest.fixture(scope="module")
def raw_fc() -> ForceConstants:
    return force_constants_from_siesta_run(FC_RUN / "fc_dhs_d0p01.FC", FC_RUN / "RUN.fdf")


def geometry_node() -> dict:
    return epc_artifact_node(
        "geometry",
        {
            "cell": [[2.2, -1.27, 0.0], [2.2, 1.27, 0.0], [0.0, 0.0, 29.34]],
            "species": ["C", "C"],
            "positions_sha256": "0" * 64,
            "coordinate_convention": "cartesian_ang",
        },
    )


# --------------------------------------------------------------------------- #
# The SIESTA adapter: parsing, symmetry, ASR
# --------------------------------------------------------------------------- #


def test_fc_parse_declares_displacement_and_blocks():
    parsed = read_siesta_fc(FC_RUN / "fc_dhs_d0p01.FC")
    assert parsed["displacement_ang"] == pytest.approx(0.01)
    assert parsed["row_atom_count"] == 2
    assert parsed["displaced_atom_count"] == 2
    # [displaced, direction, sign, atom, xyz]: the two signs differ by anharmonicity.
    assert parsed["signed_blocks"].shape == (2, 3, 2, 2, 3)


def test_force_constants_are_symmetric_and_carry_provenance(raw_fc):
    matrix = raw_fc.matrix[0].reshape(6, 6)
    assert np.abs(matrix - matrix.T).max() < 1e-8
    assert raw_fc.masses_amu == pytest.approx([CARBON_MASS_AMU] * 2, rel=1e-3)
    assert raw_fc.provenance["fc_displacement_ang"] == pytest.approx(0.01)
    assert len(raw_fc.provenance["fc_sha256"]) == 64
    assert raw_fc.provenance["geometry_signature"]


def test_ghost_species_have_no_mass():
    from phonon_provider import _masses_amu

    with pytest.raises(PhononContractError, match="ghost"):
        _masses_amu([6, -1])


def test_raw_and_corrected_ifc_are_different_artifacts(raw_fc):
    corrected = raw_fc.apply_asr()
    assert raw_fc.asr_policy == ASR_RAW
    assert corrected.asr_policy == ASR_CORRECTED
    # The raw run really does violate translation, and the number survives.
    assert raw_fc.asr_residual_ev_ang2 > 1e-3
    assert corrected.asr_residual_ev_ang2 < 1e-10
    assert corrected.asr_residual_raw_ev_ang2 == raw_fc.asr_residual_ev_ang2
    assert corrected.apply_asr() is corrected  # idempotent
    # Correcting only moves the on-site blocks.
    assert np.array_equal(raw_fc.matrix[0][0, :, 1, :], corrected.matrix[0][0, :, 1, :])
    assert not np.array_equal(raw_fc.matrix[0][0, :, 0, :], corrected.matrix[0][0, :, 0, :])


def test_gamma_frequencies_reproduce_graphene(raw_fc):
    modes = SiestaFcPhononProvider(raw_fc).modes()
    frequencies = modes.frequencies_ev
    assert modes.mode_count == 6
    assert modes.asr_policy == ASR_CORRECTED
    # Three acoustic modes, exactly zero because the sum rule was imposed.
    assert np.abs(frequencies[:3]).max() < 1e-6
    # ZO around 850 cm^-1 and the E2g doublet near 1580 cm^-1 (Piscanec 2004).
    assert 800.0 < frequencies[3] / CM1_TO_EV < 900.0
    assert frequencies[4] / CM1_TO_EV == pytest.approx(frequencies[5] / CM1_TO_EV, rel=1e-4)
    assert 1500.0 < frequencies[5] / CM1_TO_EV < 1620.0


def test_raw_ifc_leaves_the_acoustic_modes_measurably_nonzero(raw_fc):
    raw_modes = SiestaFcPhononProvider(raw_fc, apply_asr=False).modes()
    assert raw_modes.asr_policy == ASR_RAW
    assert np.abs(raw_modes.frequencies_ev[:3]).max() > 1e-3  # ~40-70 cm^-1 of noise


def test_acoustic_modes_are_uniform_translations(raw_fc):
    modes = SiestaFcPhononProvider(raw_fc).modes()
    acoustic = modes.eigenvectors[:3].reshape(3, -1)
    translation = np.zeros(6)
    translation[0::3] = 1.0 / np.sqrt(2)  # equal masses: e is the uniform x shift
    projected = acoustic.conj().T @ (acoustic @ translation)
    assert np.linalg.norm(projected - translation) < 1e-6


def test_zero_point_amplitude_is_the_formalism_constant(raw_fc):
    mode = SiestaFcPhononProvider(raw_fc).modes().mode(5)
    amplitude = mode.zero_point_amplitude_ang
    assert amplitude == pytest.approx(
        [zero_point_amplitude_ang(m, mode.frequency_ev) for m in mode.masses_amu]
    )
    assert 0.02 < amplitude[0] < 0.05  # ~0.03 Ang for carbon at 1580 cm^-1
    # displacement_pattern is that amplitude times the unit eigenvector.
    pattern = mode.displacement_pattern()
    assert np.linalg.norm(pattern) == pytest.approx(amplitude[0], rel=1e-12)
    # The acoustic amplitude diverges and is refused rather than reported.
    with pytest.raises(PhononContractError, match="diverges"):
        _ = SiestaFcPhononProvider(raw_fc).modes().mode(0).zero_point_amplitude_ang


def test_nc_factor_is_the_only_source_of_1_over_sqrt_nc():
    assert nc_factor(4) == pytest.approx(0.5)
    with pytest.raises(PhononContractError):
        nc_factor(0)
    assert "1/N_q" in CANONICAL_CONVENTIONS.nc_placement  # memo section 8: not in g


# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #


def test_atomic_phase_is_a_diagonal_unitary_on_the_dynamical_matrix(raw_fc):
    q = np.array([0.25, 0.1, 0.0])
    cell_phase = raw_fc.dynamical_matrix(q, CANONICAL_CONVENTIONS)
    atom_phase = raw_fc.dynamical_matrix(q, PhononConventions(atomic_phase=ATOMIC_PHASE_ATOM))
    reciprocal = 2.0 * np.pi * np.linalg.inv(raw_fc.cell_ang).T
    tau = np.exp(1j * (raw_fc.positions_ang @ (q @ reciprocal)))
    unitary = np.repeat(tau, 3)
    assert np.abs(atom_phase - unitary.conj()[:, None] * cell_phase * unitary[None, :]).max() < 1e-12


def test_conventions_convert_to_the_canon_without_changing_the_displacement(raw_fc):
    """The physical displacement is convention independent; the numbers are not.

    Built from *one* diagonalisation and re-expressed by hand, so the comparison
    is exact: two independent ``eigh`` calls would only agree up to the gauge of
    the degenerate acoustic and E2g blocks.
    """
    q = np.array([0.25, 0.1, 0.0])
    in_cell_phase = modes_from_force_constants(raw_fc.apply_asr(), q, provider="test")
    reciprocal = 2.0 * np.pi * np.linalg.inv(raw_fc.cell_ang).T
    tau = np.exp(1j * (raw_fc.positions_ang @ (q @ reciprocal)))
    in_atom_phase = dataclasses.replace(
        in_cell_phase,
        eigenvectors=in_cell_phase.eigenvectors * tau.conj()[None, :, None],
        conventions=PhononConventions(atomic_phase=ATOMIC_PHASE_ATOM),
    )

    converted = to_canonical_conventions(in_atom_phase)
    assert converted.conventions.atomic_phase == ATOMIC_PHASE_CELL
    assert not np.allclose(in_atom_phase.eigenvectors, in_cell_phase.eigenvectors)
    assert np.allclose(converted.eigenvectors, in_cell_phase.eigenvectors, atol=1e-14)
    for branch in range(3, converted.mode_count):  # optical: omega > 0
        assert np.allclose(
            in_atom_phase.mode(branch).displacement_pattern((1, 0, 0)),
            in_cell_phase.mode(branch).displacement_pattern((1, 0, 0)),
            atol=1e-14,
        )


def test_fourier_sign_conversion_is_a_conjugation_and_round_trips(raw_fc):
    q = (0.25, 0.1, 0.0)
    negative = modes_from_force_constants(
        raw_fc.apply_asr(), q, provider="test", conventions=PhononConventions(fourier_sign=-1)
    )
    canonical = to_canonical_conventions(negative)
    assert canonical.conventions.fourier_sign == 1
    assert np.allclose(canonical.eigenvectors, negative.eigenvectors.conj())
    # Involutive: converting what is already canonical changes nothing.
    assert to_canonical_conventions(canonical) is canonical
    positive = modes_from_force_constants(raw_fc.apply_asr(), q, provider="test")
    assert np.allclose(np.sort(canonical.frequencies_ev), np.sort(positive.frequencies_ev))


def test_normalization_is_enforced_not_assumed(raw_fc):
    modes = SiestaFcPhononProvider(raw_fc).modes()
    assert modes.conventions.eigenvector_normalization == MASS_WEIGHTED_UNIT_NORM
    with pytest.raises(PhononContractError, match="norm"):
        PhononModeSet(
            q_fractional=(0, 0, 0),
            frequencies_ev=[0.1],
            eigenvectors=np.full((1, 2, 3), 0.5),
            masses_amu=modes.masses_amu,
            cell_ang=modes.cell_ang,
            positions_ang=modes.positions_ang,
            conventions=CANONICAL_CONVENTIONS,
            provider="test",
            provider_version="1",
            force_source={},
            primitive_mapping={},
            fc_range={},
            asr_policy=ASR_RAW,
            asr_residual_ev_ang2=0.0,
            asr_residual_raw_ev_ang2=0.0,
        )
    with pytest.raises(PhononContractError):
        PhononConventions(fourier_sign=0)
    with pytest.raises(PhononContractError):
        PhononConventions(atomic_phase="whatever")


# --------------------------------------------------------------------------- #
# Round trip and artifact identity
# --------------------------------------------------------------------------- #


def test_mode_artifact_round_trips_through_json(raw_fc):
    modes = SiestaFcPhononProvider.from_run_dir(FC_RUN).modes((0.25, 0.1, 0.0))
    payload = json.loads(json.dumps(modes.to_dict()))
    restored = PhononModeSet.from_dict(payload)
    assert restored.q_fractional.tolist() == modes.q_fractional.tolist()
    assert restored.masses_amu.tolist() == modes.masses_amu.tolist()
    assert np.array_equal(restored.eigenvectors, modes.eigenvectors)  # phases survive
    assert restored.conventions == modes.conventions
    assert restored.asr_policy == modes.asr_policy
    assert restored.asr_residual_raw_ev_ang2 == modes.asr_residual_raw_ev_ang2
    assert restored.eigenvector_sha256 == modes.eigenvector_sha256
    assert payload["units"] == UNITS
    assert payload["contract_id"] == CONTRACT_ID
    assert restored.geometry_signature == modes.geometry_signature


def test_mode_set_signs_a_physical_phonon_node(raw_fc):
    modes = SiestaFcPhononProvider(raw_fc, geometry_signature="abc").modes()
    node = epc_artifact_node(
        PHYSICAL_PHONON, modes.physical_phonon_fields(), {"geometry": geometry_node()}
    )
    assert node["kind"] == PHYSICAL_PHONON
    assert SYNTHETIC_TEST_DISPLACEMENT not in node["ancestor_kinds"]
    # The ASR policy and the raw residual are part of the hash, not a comment.
    other = SiestaFcPhononProvider(raw_fc, apply_asr=False).modes()
    assert (
        epc_artifact_node(
            PHYSICAL_PHONON, other.physical_phonon_fields(), {"geometry": geometry_node()}
        )["signature_sha256"]
        != node["signature_sha256"]
    )


def test_synthetic_displacement_cannot_become_a_phonon(raw_fc):
    synthetic = SyntheticTestDisplacement(
        label="uniform_shear_like", displacement_ang=np.eye(2, 3) / np.sqrt(2)
    )
    assert synthetic.artifact_kind == SYNTHETIC_TEST_DISPLACEMENT
    assert not hasattr(synthetic, "frequencies_ev")

    with pytest.raises(PhononContractError, match="roadmap B7"):
        require_physical_phonon(synthetic)
    with pytest.raises(PhononContractError):
        PhononModeSet.from_dict(synthetic.to_dict())
    # And the DAG refuses it as a physical_phonon node too.
    with pytest.raises(EpcDagError):
        epc_artifact_node(
            PHYSICAL_PHONON, synthetic.artifact_fields(), {"geometry": geometry_node()}
        )
    node = epc_artifact_node(
        SYNTHETIC_TEST_DISPLACEMENT, synthetic.artifact_fields(), {"geometry": geometry_node()}
    )
    assert node["kind"] == SYNTHETIC_TEST_DISPLACEMENT

    modes = SiestaFcPhononProvider(raw_fc).modes()
    assert require_physical_phonon(modes) is modes


# --------------------------------------------------------------------------- #
# VIBRA adapter, and the core that must not know about it
# --------------------------------------------------------------------------- #


def _vibra_vectors_text(modes: PhononModeSet, k_1_bohr) -> str:
    """A VIBRA ``.vectors`` file for a mode set already in the atom-phase convention."""
    lines = ["", "k            = " + " ".join(f"{value:12.6f}" for value in k_1_bohr)]
    for branch in range(modes.mode_count):
        lines.append(f"Eigenvector  = {branch + 1:5d}")
        lines.append(f"Frequency    = {modes.frequencies_ev[branch] / CM1_TO_EV:12.6f}")
        for part in ("real", "imag"):
            lines.append(f"Eigenmode ({part if part == 'real' else 'imaginary'} part)")
            values = getattr(modes.eigenvectors[branch], part)
            lines.extend("  ".join(f"{value:12.6f}" for value in row) for row in values)
    return "\n".join(lines) + "\n"


def test_vibra_adapter_converts_its_own_conventions(raw_fc, tmp_path):
    q = np.array([0.25, 0.1, 0.0])
    corrected = raw_fc.apply_asr()
    atom_phase = modes_from_force_constants(
        corrected, q, provider="reference", conventions=PhononConventions(atomic_phase=ATOMIC_PHASE_ATOM)
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(raw_fc.cell_ang).T
    from fdf_materialization import BOHR_TO_ANG

    vectors = tmp_path / "graphene.vectors"
    vectors.write_text(_vibra_vectors_text(atom_phase, (q @ reciprocal) * BOHR_TO_ANG))

    provider = VibraPhononProvider.from_vectors_file(vectors, corrected)
    modes = provider.modes(q)

    assert modes.conventions.is_canonical  # converted at the adapter boundary
    assert modes.provider == "siesta_vibra"
    assert modes.asr_policy == ASR_CORRECTED
    assert modes.asr_residual_raw_ev_ang2 == raw_fc.asr_residual_ev_ang2
    # Compared against the very modes the file was written from, so any gauge of
    # the degenerate blocks is common to both: what is measured here is the
    # parsing, the cm^-1 -> eV conversion and the atomic-phase removal.
    reference = to_canonical_conventions(atom_phase)
    assert np.allclose(modes.frequencies_ev, reference.frequencies_ev, atol=1e-6)
    assert np.allclose(modes.eigenvectors, reference.eigenvectors, atol=1e-5)
    with pytest.raises(PhononContractError, match="not in the file"):
        provider.modes((0.5, 0.5, 0.0))


def test_epc_core_contains_no_vibra_parsing():
    """Acceptance: the backend-specific parsing lives in the adapter, not the core."""
    core = ("epc_formalism.py", "epc_subspaces.py", "epc_occupations.py", "epc_basis_response.py")
    for name in core:
        text = (REPO_ROOT / "Comparison" / "scripts" / name).read_text(encoding="utf-8")
        assert "vibra" not in text.lower(), f"{name} parses a backend the contract hides"
