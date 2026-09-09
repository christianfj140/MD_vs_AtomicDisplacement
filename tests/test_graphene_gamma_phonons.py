"""C18 / E-F_001-S21: Gamma phonons of graphene at two auxiliary FC ranges.

The campaign itself needs SIESTA, so what is checked without it is everything
that decides whether its numbers mean anything:

* the supercell mapping. ``Phi[l, kappa, alpha, kappa', beta]`` is read out of a
  flat ``.FC`` by index arithmetic, and an off-by-one there would silently
  produce plausible frequencies. It is checked by writing a ``.FC`` whose force
  constants are known and asserting the exact reconstruction, and by the physics
  the campaign rests on: ``D(Gamma) = sum_l Phi_l`` is what a 1x1 run already
  computes implicitly;
* the sector identification. Acoustic and E2g are read off the eigenvectors, so
  the test rotates the gauge inside each degenerate block and demands the same
  answer -- an identification that survives a random unitary was measured, not
  imposed;
* the fdf layer that goes to SIESTA, because a stray row from a dropped block is
  a silently different calculation.

The produced campaign, when it is present, is checked as an artifact: two ranges,
raw and ASR-corrected modes as separate files, and the backend recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import phonon_provider as pp  # noqa: E402
import run_graphene_gamma_phonons as campaign  # noqa: E402

FC_RUN = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene/runs/fc_dhs_d0p01"
MANIFEST = campaign.DEFAULT_OUTPUT_ROOT / campaign.MANIFEST_NAME

CELL = np.array([[2.2005, -1.2704592674, 0.0], [2.2005, 1.2704592674, 0.0], [0.0, 0.0, 29.34]])
POSITIONS = np.array([[1.467, 0.0, 0.0], [2.934, 0.0, 0.0]])
MASSES = np.array([12.0107, 12.0107])


def write_fc_file(path: Path, matrix: np.ndarray, *, displacement_ang: float = 0.01) -> Path:
    """A ``.FC`` in SIESTA's own layout from ``[displaced, 3, supercell atom, 3]``."""
    n_displaced, _, n_rows, _ = matrix.shape
    lines = [f"Force constants matrix. n_atoms, displacement [Ang]: {n_rows} {displacement_ang:.16E}"]
    for displaced in range(n_displaced):
        for axis in range(3):
            for _sign in range(2):  # both signed blocks; their mean is the IFC
                for row in range(n_rows):
                    lines.append("  ".join(f"{value:.9E}" for value in matrix[displaced, axis, row]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def random_force_constants(images: np.ndarray, seed: int = 0) -> np.ndarray:
    """A symmetric-by-construction ``[n_cells, 2, 3, 2, 3]`` toy IFC."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(images.shape[0], 2, 3, 2, 3))


# --------------------------------------------------------------------------- #
# The supercell mapping
# --------------------------------------------------------------------------- #


def test_supercell_images_are_symmetric_and_start_at_the_origin():
    images = pp.supercell_images((3, 3, 1))
    assert images.shape == (9, 3)
    assert list(images[0]) == [0, 0, 0]
    assert {tuple(image) for image in images} == {(i, j, 0) for i in (-1, 0, 1) for j in (-1, 0, 1)}
    assert list(pp.supercell_images((1, 1, 1))[0]) == [0, 0, 0]
    for bad in ((2, 2, 1), (3, 3), (0, 1, 1)):
        with pytest.raises(pp.PhononContractError):
            pp.supercell_images(bad)


def test_supercell_geometry_is_image_major_with_the_unit_cell_first():
    images = pp.supercell_images((3, 3, 1))
    positions, lattice = pp.supercell_geometry(POSITIONS, CELL, images)
    assert positions.shape == (18, 3)
    assert np.allclose(positions[:2], POSITIONS)  # FC.First 1 / FC.Last 2 displaces these
    assert np.allclose(lattice, np.diag([3, 3, 1]) @ CELL)
    for index, image in enumerate(images):
        for atom in range(2):
            assert np.allclose(positions[index * 2 + atom], POSITIONS[atom] + image @ CELL)


def test_supercell_fc_reconstructs_the_force_constants_it_was_written_from(tmp_path):
    images = pp.supercell_images((3, 3, 1))
    expected = random_force_constants(images)
    # [displaced, 3, supercell atom, 3] is what SIESTA writes; supercell atom is
    # l*na + kappa' by construction of supercell_geometry.
    flat = expected.transpose(1, 2, 0, 3, 4).reshape(2, 3, 18, 3)
    fc_path = write_fc_file(tmp_path / "sc3x3x1.FC", flat)

    force_constants = pp.force_constants_from_siesta_supercell_run(
        fc_path,
        images=images,
        cell_ang=CELL,
        positions_ang=POSITIONS,
        masses_amu=MASSES,
        provenance={"note": "synthetic"},
    )
    assert np.allclose(force_constants.matrix, expected)
    assert force_constants.cell_count == 9
    assert force_constants.asr_policy == pp.ASR_RAW
    assert force_constants.provenance["supercell_repeats"] == [3, 3, 1]
    assert force_constants.provenance["fc_displacement_ang"] == pytest.approx(0.01)


def test_a_supercell_fc_with_the_wrong_row_count_is_refused(tmp_path):
    images = pp.supercell_images((3, 3, 1))
    fc_path = write_fc_file(tmp_path / "wrong.FC", np.zeros((2, 3, 8, 3)))
    with pytest.raises(pp.PhononContractError, match="force rows"):
        pp.force_constants_from_siesta_supercell_run(
            fc_path, images=images, cell_ang=CELL, positions_ang=POSITIONS, masses_amu=MASSES
        )


def test_gamma_only_sums_the_images_which_is_why_ranges_are_comparable():
    """``D(Gamma) = sum_l Phi_l``: the claim the whole campaign rests on."""
    images = pp.supercell_images((3, 3, 1))
    matrix = random_force_constants(images)
    wide = pp.ForceConstants(
        matrix=matrix, cell_images=images, cell_ang=CELL, positions_ang=POSITIONS, masses_amu=MASSES
    )
    folded = pp.ForceConstants(
        matrix=matrix.sum(axis=0)[None],
        cell_images=[[0, 0, 0]],
        cell_ang=CELL,
        positions_ang=POSITIONS,
        masses_amu=MASSES,
    )
    assert np.allclose(wide.dynamical_matrix(), folded.dynamical_matrix())
    assert np.allclose(wide.asr_violation, folded.asr_violation)
    # Away from Gamma they are different objects, which is why a 1x1 run is a
    # Gamma artifact only.
    assert not np.allclose(wide.dynamical_matrix((0.25, 0.1, 0.0)), folded.dynamical_matrix((0.25, 0.1, 0.0)))


# --------------------------------------------------------------------------- #
# The IFC tail
# --------------------------------------------------------------------------- #


def test_ifc_shells_are_distance_ordered_and_measure_the_tail():
    images = pp.supercell_images((3, 3, 1))
    matrix = random_force_constants(images)
    # A decaying IFC: scale each image block by 1/(1+|R|^2).
    lattice = images @ CELL
    for index in range(images.shape[0]):
        matrix[index] /= 1.0 + np.linalg.norm(lattice[index]) ** 2
    shells = pp.ifc_shells(
        pp.ForceConstants(
            matrix=matrix, cell_images=images, cell_ang=CELL, positions_ang=POSITIONS, masses_amu=MASSES
        )
    )
    distances = [shell["distance_ang"] for shell in shells["shells"]]
    assert distances == sorted(distances)
    assert distances[0] == 0.0
    assert shells["max_distance_ang"] == pytest.approx(max(distances))
    assert shells["cell_count"] == 9
    assert 0.0 < shells["tail_ratio"] < 1.0
    assert sum(shell["block_count"] for shell in shells["shells"]) == 9 * 2 * 2


# --------------------------------------------------------------------------- #
# Sectors: measured, not imposed
# --------------------------------------------------------------------------- #


pytestmark_fc = pytest.mark.skipif(not FC_RUN.is_dir(), reason="graphene FC run not archived here")


@pytest.fixture(scope="module")
def gamma_modes() -> pp.PhononModeSet:
    force_constants = pp.force_constants_from_siesta_supercell_run(
        FC_RUN / "fc_dhs_d0p01.FC",
        images=pp.supercell_images((1, 1, 1)),
        cell_ang=CELL,
        positions_ang=POSITIONS,
        masses_amu=pp.masses_amu_from_fdf(FC_RUN / "RUN.fdf"),
    )
    return pp.SiestaFcPhononProvider(force_constants).modes()


@pytestmark_fc
def test_the_supercell_reader_agrees_with_the_unit_cell_reader():
    """One image is the 1x1 case; the two builders must not disagree about it."""
    direct = pp.force_constants_from_siesta_run(FC_RUN / "fc_dhs_d0p01.FC", FC_RUN / "RUN.fdf")
    through_images = pp.force_constants_from_siesta_supercell_run(
        FC_RUN / "fc_dhs_d0p01.FC",
        images=pp.supercell_images((1, 1, 1)),
        cell_ang=direct.cell_ang,
        positions_ang=direct.positions_ang,
        masses_amu=direct.masses_amu,
    )
    assert np.array_equal(direct.matrix, through_images.matrix)


@pytestmark_fc
def test_sectors_are_identified_from_the_eigenvectors(gamma_modes):
    report = pp.mode_sector_report(gamma_modes)
    assert report["acoustic_branches"] == [0, 1, 2]
    assert report["identification"]["acoustic_unambiguous"]
    e2g = report["e2g_candidate"]
    assert e2g is not None and e2g["branches"] == [4, 5]
    assert 1500.0 < e2g["mean_frequency_cm1"] < 1620.0  # Piscanec 2004
    assert report["identification"]["e2g_degeneracy_resolved_as_one_cluster"]
    # ZO: optical, out of plane, and therefore not the E2g doublet.
    zo = report["branches"][3]
    assert 800.0 < zo["frequency_cm1"] < 900.0
    assert zo["out_of_plane_weight"] > 0.99
    assert zo["translation_weight"] < 0.01


@pytestmark_fc
def test_sector_identification_survives_a_gauge_rotation_of_each_degenerate_block(gamma_modes):
    """A random unitary inside the acoustic and E2g blocks changes no conclusion."""
    import dataclasses

    rng = np.random.default_rng(11)
    eigenvectors = gamma_modes.eigenvectors.copy()
    for block in ([0, 1, 2], [4, 5]):
        size = len(block)
        unitary = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))[0]
        eigenvectors[block] = np.tensordot(unitary, eigenvectors[block], axes=(1, 0))
    rotated = dataclasses.replace(gamma_modes, eigenvectors=eigenvectors)

    before = pp.mode_sector_report(gamma_modes)
    after = pp.mode_sector_report(rotated)
    assert not np.allclose(rotated.eigenvectors, gamma_modes.eigenvectors)
    for key in ("acoustic_count", "acoustic_unambiguous", "e2g_identified", "e2g_degeneracy_resolved_as_one_cluster"):
        assert after["identification"][key] == before["identification"][key]
    assert after["acoustic_branches"] == before["acoustic_branches"]
    assert after["e2g_candidate"]["branches"] == before["e2g_candidate"]["branches"]
    assert after["e2g_candidate"]["mean_frequency_cm1"] == pytest.approx(
        before["e2g_candidate"]["mean_frequency_cm1"]
    )
    # Per branch the translation weight is invariant (the whole acoustic block
    # is the translation subspace); the in/out-of-plane split is only invariant
    # per cluster, because mixing ZA into LA/TA is exactly what a gauge does.
    for left, right in zip(before["branches"], after["branches"]):
        assert left["translation_weight"] == pytest.approx(right["translation_weight"], abs=1e-10)
    for left, right in zip(before["clusters"], after["clusters"]):
        assert left["branches"] == right["branches"]
        assert left["out_of_plane_weight"] == pytest.approx(right["out_of_plane_weight"], abs=1e-10)


@pytestmark_fc
def test_raw_and_corrected_gamma_modes_are_different_artifacts():
    force_constants = pp.force_constants_from_siesta_run(FC_RUN / "fc_dhs_d0p01.FC", FC_RUN / "RUN.fdf")
    raw = pp.SiestaFcPhononProvider(force_constants, apply_asr=False).modes()
    corrected = pp.SiestaFcPhononProvider(force_constants).modes()
    assert raw.asr_policy == pp.ASR_RAW and corrected.asr_policy == pp.ASR_CORRECTED
    # The raw acoustic modes are measurably non-zero and the residual survives
    # the correction, so "omega = 0" is never a claim the correction hid.
    assert np.abs(raw.frequencies_ev[:3]).max() > 1e-4
    assert np.abs(corrected.frequencies_ev[:3]).max() < 1e-6
    assert corrected.asr_residual_raw_ev_ang2 == raw.asr_residual_ev_ang2 > 0.0
    assert pp.mode_sector_report(raw)["acoustic_branches"] == [0, 1, 2]


def test_frequency_resolution_comes_from_the_noise_floor():
    assert np.allclose(pp.frequency_resolutions_ev([0.2, 0.0], 0.0), 0.0)
    resolution = pp.frequency_resolutions_ev([0.2], 1e-3)[0]
    shift = pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV * 1e-3
    assert resolution == pytest.approx(np.sqrt(0.2**2 + shift) - 0.2)
    # A branch at zero is resolved only to sqrt(C*dlambda), not to zero.
    assert pp.frequency_resolutions_ev([0.0], 1e-3)[0] == pytest.approx(np.sqrt(shift))


# --------------------------------------------------------------------------- #
# The fdf that reaches SIESTA
# --------------------------------------------------------------------------- #


def test_analysis_layer_is_dropped_whole():
    text = campaign.DEFAULT_MATERIAL_FDF.read_text(encoding="utf-8")
    stripped, removed = campaign.drop_analysis_layer(text)
    # Comments survive and are irrelevant; what SIESTA reads is the directives.
    lowered = "\n".join(line.split("#", 1)[0] for line in stripped.splitlines()).lower()
    for token in ("bandlines", "projecteddensityofstates", "wannier", "%pdos", "writemullikenpop"):
        assert token not in lowered
    # No orphan row of a dropped block may survive as a directive.
    assert "60  0  0  0.5" not in stripped
    assert "1   0.0       0.0        0.0   \\Gamma" not in stripped
    # Everything that decides a force is untouched.
    for kept in ("MeshCutoff", "XC.authors", "%block PAO.Basis", "ElectronicTemperature", "DM.Tolerance"):
        assert kept in stripped
    assert removed


def test_kgrid_is_scaled_by_the_repeats():
    assert campaign.scaled_kgrid((1, 1, 1), [20, 20, 1]) == [20, 20, 1]
    assert campaign.scaled_kgrid((3, 3, 1), [20, 20, 1]) == [7, 7, 1]  # ceil: never sparser
    assert campaign.scaled_kgrid((5, 5, 1), [20, 20, 1]) == [4, 4, 1]
    assert campaign.base_kgrid(campaign.DEFAULT_MATERIAL_FDF.read_text(encoding="utf-8")) == [20, 20, 1]


def test_a_single_range_is_refused():
    assert campaign.parse_ranges("1x1x1,3x3x1") == [[1, 1, 1], [3, 3, 1]]
    with pytest.raises(campaign.GammaPhononError, match="at least two"):
        campaign.parse_ranges("3x3x1")


def test_range_signature_covers_geometry_basis_pseudos_and_fc_settings():
    staged = {
        "repeats": [3, 3, 1],
        "cell_images": [[0, 0, 0]],
        "kgrid_monkhorst_pack": [7, 7, 1],
        "materialized_fdf_sha256": "a" * 64,
        "physics_fingerprint": "b" * 64,
        "pseudopotential_sha256": {"C.psf": "c" * 64},
        "fc_directives": {"displacement_ang": 0.01, "first_atom": 1, "last_atom": 2, "save_dhs": False},
    }
    shared = {"base_fdf_sha256": "d" * 64, "basis_contract_hash": "e" * 64, "siesta_binary_sha256": "f" * 64}
    reference = campaign.range_signature(staged, shared)
    assert campaign.range_signature(dict(staged), dict(shared)) == reference
    for mutate in (
        lambda: staged.__setitem__("kgrid_monkhorst_pack", [8, 8, 1]),
        lambda: staged["fc_directives"].__setitem__("displacement_ang", 0.02),
        lambda: staged["pseudopotential_sha256"].__setitem__("C.psf", "0" * 64),
        lambda: shared.__setitem__("basis_contract_hash", "0" * 64),
        lambda: shared.__setitem__("siesta_binary_sha256", "0" * 64),
    ):
        snapshot = json.dumps([staged, shared], sort_keys=True)
        mutate()
        assert campaign.range_signature(staged, shared) != reference
        staged_restored, shared_restored = json.loads(snapshot)
        staged.update(staged_restored)
        shared.update(shared_restored)


def test_compare_ranges_refuses_to_compare_different_branch_counts():
    def row(name, frequencies):
        modes = {
            policy: {
                "frequencies_cm1": frequencies,
                "sector_report": {"e2g_candidate": {"mean_frequency_cm1": frequencies[-1]}},
                "ifc_shells": {"tail_ratio": 0.5, "max_distance_ang": 1.4},
            }
            for policy in ("raw", "asr_corrected")
        }
        return {"range": name, "certified": True, "modes": modes}

    comparison = campaign.compare_ranges([row("sc1x1x1", [0.0, 1580.0]), row("sc3x3x1", [0.0, 1575.0])])
    assert comparison["at_least_two_ranges"]
    assert comparison["largest_max_abs_delta_cm1"] == pytest.approx(5.0)
    assert comparison["steps"][0]["e2g_delta_cm1"] == pytest.approx(-5.0)
    with pytest.raises(campaign.GammaPhononError, match="branches"):
        campaign.compare_ranges([row("a", [0.0, 1580.0]), row("b", [0.0, 1.0, 1575.0])])


# --------------------------------------------------------------------------- #
# The produced campaign
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not MANIFEST.is_file(), reason="the Gamma phonon campaign has not been run here")
def test_produced_campaign_publishes_two_ranges_raw_and_corrected():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == campaign.SCHEMA
    assert manifest["ranges_failed"] == 0
    assert manifest["range_comparison"]["at_least_two_ranges"]
    assert manifest["q_fractional"] == [0.0, 0.0, 0.0]
    # The compute policy is recorded with its evidence, not asserted.
    assert manifest["compute_policy"]["effective_backend"] == "cpu"
    assert manifest["compute_policy"]["reason"]

    for row in manifest["rows"]:
        assert row["certified"], row.get("problems")
        assert row["scf_convergence"]["all_steps_converged"]
        assert row["wall_seconds"] >= 0.0
        raw, corrected = row["modes"]["raw"], row["modes"]["asr_corrected"]
        assert raw["asr_policy"] == pp.ASR_RAW
        assert corrected["asr_policy"] == pp.ASR_CORRECTED
        assert raw["signature_sha256"] != corrected["signature_sha256"]
        assert corrected["asr_residual_raw_ev_ang2"] == raw["asr_residual_ev_ang2"]
        assert corrected["sector_report"]["identification"]["acoustic_unambiguous"]
        e2g = corrected["sector_report"]["e2g_candidate"]
        assert e2g is not None and 1500.0 < e2g["mean_frequency_cm1"] < 1620.0
        for policy in ("raw", "asr_corrected"):
            path = REPO_ROOT / row["modes"][policy]["path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["artifact_kind"] == pp.PHYSICAL_PHONON
            assert pp.PhononModeSet.from_dict(payload).mode_count == 6
