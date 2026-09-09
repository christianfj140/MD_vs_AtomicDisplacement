"""C03 / E-F_001-S4: the basis/orbital compatibility gate.

Every test runs on the repository's real graphene / AB / MATBG inputs or on a
fixture derived from them. Nothing here runs SIESTA, Graph2Mat or a prediction;
the MATBG counts are obtained by counting the FDF and the .ion.xml.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from orbital_contract import (  # noqa: E402
    DECLARED_ONLY,
    INVALID,
    VERIFIED,
    build_orbital_contract,
    certify_system,
    certify_systems,
    compare_contracts,
    graph2mat_basis_tuple,
    parse_orb_indx,
    read_ion_xml,
)

GRAPHENE_FDF = REPO_ROOT / "materials/graphene/RUN.fdf"
GRAPHENE_BASIS = REPO_ROOT / "materials/graphene/basis"
COMMON_BASIS = REPO_ROOT / "materials/graphene_common/basis"
AB_FDF = REPO_ROOT / "materials/bilayer_graphene_AB/RUN.fdf"
MATBG_FDF = REPO_ROOT / "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf"


def _outcome(record: dict, check: str) -> str:
    return next(item["outcome"] for item in record["checks"] if item["check"] == check)


def _graphene_orb_indx() -> Path:
    matches = sorted(REPO_ROOT.glob("Comparison/datasets/graphene_w90_snapshot_scaling/**/graphene.ORB_INDX"))
    if not matches:
        pytest.skip("no graphene ORB_INDX in this checkout")
    return matches[0]


# ---------------------------------------------------------------------------
# .ion.xml semantics
# ---------------------------------------------------------------------------


def test_carbon_ion_xml_expands_to_the_siesta_orbital_order():
    carbon = read_ion_xml(COMMON_BASIS / "C.ion.xml")
    assert carbon["atomic_number"] == 6
    assert carbon["is_ghost"] is False
    assert carbon["orbitals_per_atom"] == 4
    # 2s, then the l=1 shell expanded m = -1, 0, +1 (py, pz, px).
    assert [(o["n"], o["l"], o["m"], o["zeta"]) for o in carbon["orbitals"]] == [
        (2, 0, 0, 1),
        (2, 1, -1, 1),
        (2, 1, 0, 1),
        (2, 1, 1, 1),
    ]
    assert graph2mat_basis_tuple(carbon["shells"]) == ((1, 0, 1), (1, 1, -1))


def test_ghost_species_is_recognised_by_its_negative_z():
    ghost = read_ion_xml(GRAPHENE_BASIS / "Ghost-H.ion.xml")
    assert ghost["atomic_number"] == -1
    assert ghost["is_ghost"] is True
    # Graph2Mat/sisl only see |Z|, so a ghost H and a real H share a point type.
    assert ghost["graph2mat_point_type"] == 1
    assert ghost["orbitals_per_atom"] == 1


def test_multizeta_polarised_species_group_by_l():
    nitrogen = read_ion_xml(REPO_ROOT / "materials/bn/basis/N.ion.xml")
    assert nitrogen["orbitals_per_atom"] == 13  # 2s + 2s + 3p + 3p + 5d
    assert graph2mat_basis_tuple(nitrogen["shells"]) == ((2, 0, 1), (2, 1, -1), (1, 2, 1))
    assert [o["polarized"] for o in nitrogen["orbitals"]][-5:] == [True] * 5


# ---------------------------------------------------------------------------
# Ghost-H declared in graphene
# ---------------------------------------------------------------------------


def test_graphene_ghost_h_is_declared_but_contributes_no_orbital():
    contract = build_orbital_contract("graphene", GRAPHENE_FDF, GRAPHENE_BASIS)
    assert [record["label"] for record in contract["species"]] == ["C"]
    assert [record["label"] for record in contract["declared_unpopulated_species"]] == ["Ghost-H"]
    assert contract["atom_count"] == 2
    assert contract["orbital_count"] == 8


def test_ghost_basis_file_is_reported_as_an_extra_graph2mat_point_type():
    record = certify_system("graphene", GRAPHENE_FDF, GRAPHENE_BASIS)
    detail = next(item for item in record["checks"] if item["check"] == "unpopulated_species_Ghost-H")["detail"]
    assert "0 atoms" in detail
    assert "point type 1" in detail


def test_graphene_basis_dir_and_common_basis_dir_agree_on_carbon():
    with_ghost = build_orbital_contract("graphene", GRAPHENE_FDF, GRAPHENE_BASIS)
    without_ghost = build_orbital_contract("graphene", GRAPHENE_FDF, COMMON_BASIS)
    # The ghost has no atoms, so it is outside the basis contract either way.
    assert with_ghost["basis_contract_hash"] == without_ghost["basis_contract_hash"]


# ---------------------------------------------------------------------------
# ORB_INDX certification
# ---------------------------------------------------------------------------


def test_orb_indx_certifies_order_cutoffs_and_periodic_folding():
    record = certify_system("graphene", GRAPHENE_FDF, GRAPHENE_BASIS, orb_indx=_graphene_orb_indx())
    for check in (
        "orbital_count_matches_orb_indx",
        "orbital_order_matches_orb_indx",
        "orbital_cutoffs_match_orb_indx",
        "supercell_image_folding",
    ):
        assert _outcome(record, check) == "pass", check
    assert record["orb_indx"]["rc_units"] == "Bohr"
    assert record["orb_indx"]["supercell_images"] * record["orb_indx"]["orbitals_unit_cell"] == (
        record["orb_indx"]["orbitals_supercell"]
    )


def test_swapping_the_p_order_in_orb_indx_invalidates_the_contract(tmp_path):
    source = _graphene_orb_indx()
    lines = source.read_text().splitlines()
    # Swap py (m=-1) and px (m=+1) of the first atom: same orbitals, wrong order.
    header, rows = lines[:2], lines[2:]
    rows[1], rows[3] = rows[3], rows[1]
    target = tmp_path / "graphene.ORB_INDX"
    target.write_text("\n".join(header + rows) + "\n")

    record = certify_system("graphene", GRAPHENE_FDF, GRAPHENE_BASIS, orb_indx=target)
    assert _outcome(record, "orbital_order_matches_orb_indx") == "fail"
    assert record["status"] == INVALID


def test_a_changed_orb_indx_table_changes_the_orbital_contract_hash(tmp_path):
    source = _graphene_orb_indx()
    reference = certify_system("graphene", GRAPHENE_FDF, GRAPHENE_BASIS, orb_indx=source)

    lines = source.read_text().splitlines()
    first_row = next(index for index, line in enumerate(lines) if len(line.split()) == 16)
    lines[first_row] = lines[first_row].replace("  4.089", "  4.500")  # a different rc for 2s
    target = tmp_path / "graphene.ORB_INDX"
    target.write_text("\n".join(lines) + "\n")

    changed = certify_system("graphene", GRAPHENE_FDF, GRAPHENE_BASIS, orb_indx=target)
    assert changed["orbital_contract_hash"] != reference["orbital_contract_hash"]
    assert _outcome(changed, "orbital_cutoffs_match_orb_indx") == "fail"
    assert not compare_contracts(reference, changed)["compatible"]


def test_parse_orb_indx_reads_the_supercell_columns():
    parsed = parse_orb_indx(_graphene_orb_indx())
    assert parsed["rows_parsed"] == parsed["orbitals_supercell"]
    assert all(row["isc"] == [0, 0, 0] for row in parsed["unit_cell_rows"])
    assert {row["iuo"] for row in parsed["rows"]} == {row["io"] for row in parsed["unit_cell_rows"]}


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_a_different_basis_file_invalidates_the_contract(tmp_path):
    reference = build_orbital_contract("graphene", GRAPHENE_FDF, COMMON_BASIS)

    basis = tmp_path / "basis"
    basis.mkdir()
    text = (COMMON_BASIS / "C.ion.xml").read_text()
    (basis / "C.ion.xml").write_text(text.replace("      4.08935569587", "      4.50000000000", 1))
    changed = build_orbital_contract("graphene", GRAPHENE_FDF, basis)

    assert changed["basis_contract_hash"] != reference["basis_contract_hash"]
    assert changed["orbital_contract_hash"] != reference["orbital_contract_hash"]
    assert compare_contracts(reference, changed)["changed"] == [
        "basis_contract_hash",
        "orbital_contract_hash",
    ]


def test_a_different_species_set_invalidates_the_contract(tmp_path):
    reference = build_orbital_contract("bilayer_AB", AB_FDF, COMMON_BASIS)

    basis = tmp_path / "basis"
    basis.mkdir()
    shutil.copy2(COMMON_BASIS / "C.ion.xml", basis / "C.ion.xml")
    shutil.copy2(REPO_ROOT / "materials/bn/basis/B.ion.xml", basis / "B.ion.xml")
    coordinates = [line for line in AB_FDF.read_text().splitlines() if line.strip().endswith("# C")]
    assert len(coordinates) == 4
    coordinates[0] = coordinates[0].replace(" 1  # C", " 2  # B")
    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(
        "NumberOfAtoms                    4\n"
        "NumberOfSpecies                  2\n"
        "%block ChemicalSpeciesLabel\n  1   6  C\n  2   5  B\n%endblock ChemicalSpeciesLabel\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n" + "\n".join(coordinates) + "\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    changed = build_orbital_contract("bilayer_AB", fdf, basis)

    assert {record["label"] for record in changed["species"]} == {"B", "C"}
    assert changed["basis_contract_hash"] != reference["basis_contract_hash"]
    assert changed["orbital_count"] != reference["orbital_count"]


def test_shells_listed_out_of_ascending_l_are_rejected(tmp_path):
    # Graph2Mat groups its blocks by l; a file that lists p before s would give
    # a block layout SIESTA never emits.
    basis = tmp_path / "basis"
    basis.mkdir()
    text = (COMMON_BASIS / "C.ion.xml").read_text()
    head, s_shell, p_shell_and_rest = text.partition("<orbital \n l=\"                        0\"")
    assert s_shell, "fixture assumes the s shell is listed first"
    s_block, _, tail = p_shell_and_rest.partition("</orbital>")
    p_block, _, rest = tail.partition("</orbital>")
    (basis / "C.ion.xml").write_text(head + p_block + "</orbital>" + s_shell + s_block + "</orbital>" + rest)

    record = certify_system("graphene", GRAPHENE_FDF, basis)
    assert _outcome(record, "shells_ascending_l_C") == "fail"
    assert record["status"] == INVALID


def test_missing_basis_file_fails_instead_of_guessing(tmp_path):
    basis = tmp_path / "basis"
    basis.mkdir()
    record = certify_system("graphene", GRAPHENE_FDF, basis)
    assert _outcome(record, "basis_file_C") == "fail"
    assert record["status"] == INVALID


# ---------------------------------------------------------------------------
# Counts without generating an H
# ---------------------------------------------------------------------------


def test_matbg_counts_are_verified_from_geometry_and_basis_only():
    record = certify_system(
        "twisted_bilayer_graphene_1p084549deg",
        MATBG_FDF,
        COMMON_BASIS,
        expected_atoms=11164,
        expected_orbitals=44656,
    )
    assert record["atom_count"] == 11164
    assert record["orbital_count"] == 44656
    assert _outcome(record, "expected_atom_count") == "pass"
    assert _outcome(record, "expected_orbital_count") == "pass"
    assert _outcome(record, "fdf_atom_count") == "pass"
    # No ORB_INDX exists for MATBG, and that must be visible, not assumed away.
    assert _outcome(record, "orb_indx_certification") == "absent"
    assert record["status"] == DECLARED_ONLY


def test_wrong_expected_orbital_count_fails():
    record = certify_system(
        "twisted_bilayer_graphene_1p084549deg",
        MATBG_FDF,
        COMMON_BASIS,
        expected_orbitals=44655,
    )
    assert _outcome(record, "expected_orbital_count") == "fail"
    assert record["status"] == INVALID


# ---------------------------------------------------------------------------
# Spin and block conventions
# ---------------------------------------------------------------------------


def test_spin_treatment_is_read_from_the_fdf_not_assumed(tmp_path):
    record = build_orbital_contract("graphene", GRAPHENE_FDF, COMMON_BASIS)
    assert record["spin"] == {
        "treatment": "non-polarized",
        "spin_components": 1,
        "declared_by": "default",
    }

    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(GRAPHENE_FDF.read_text() + "\nSpin  polarized\n")
    polarized = build_orbital_contract("graphene", fdf, COMMON_BASIS)
    assert polarized["spin"]["spin_components"] == 2
    assert polarized["basis_contract_hash"] != record["basis_contract_hash"]


def test_dotted_fdf_spelling_is_not_read_as_unpolarised(tmp_path):
    # FDF ignores case and `. - _` in labels; a gate that misses Spin.Polarized
    # would silently certify a 2-component H as a 1-component one.
    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(GRAPHENE_FDF.read_text() + "\nSpin.Polarized  .true.\n")
    spin = build_orbital_contract("graphene", fdf, COMMON_BASIS)["spin"]
    assert spin == {"treatment": "polarized", "spin_components": 2, "declared_by": "SpinPolarized"}


def test_block_conventions_are_recorded_explicitly():
    conventions = build_orbital_contract("graphene", GRAPHENE_FDF, COMMON_BASIS)["conventions"]
    assert conventions["m_order"] == "ascending_m_from_-l_to_+l_per_shell"
    assert conventions["graph2mat_basis_convention"] == "siesta_spherical"
    assert conventions["cutoff_units"] == "Bohr"
    assert "isc" in conventions["supercell_rule"]


# ---------------------------------------------------------------------------
# The cross-system statement C03 has to make
# ---------------------------------------------------------------------------


def test_graphene_ab_and_matbg_share_one_basis_contract():
    report = certify_systems(checkpoint=None)
    labels = {record["label"] for record in report["systems"]}
    assert labels == {"graphene", "bilayer_graphene_AB", "twisted_bilayer_graphene_1p084549deg"}
    assert report["basis_contract_hash"] is not None
    assert len({record["basis_contract_hash"] for record in report["systems"]}) == 1
    # Same basis, different systems: the per-system contract must still differ.
    assert len({record["orbital_contract_hash"] for record in report["systems"]}) == 3
    assert report["status"] != INVALID


def test_a_system_on_a_foreign_basis_breaks_the_shared_contract():
    report = certify_systems(
        systems=(
            {"label": "graphene", "fdf": "materials/graphene/RUN.fdf", "basis_dir": "materials/graphene/basis"},
            {"label": "hBN", "fdf": "materials/bn/RUN.fdf", "basis_dir": "materials/bn/basis"}
            if (REPO_ROOT / "materials/bn/RUN.fdf").is_file()
            else {
                "label": "graphene_on_bn_basis",
                "fdf": "materials/graphene/RUN.fdf",
                "basis_dir": "materials/bn/basis",
            },
        ),
        checkpoint=None,
    )
    assert report["basis_contract_hash"] is None
    assert report["status"] == INVALID


def test_a_system_absent_from_the_checkout_is_not_compatible_by_default():
    report = certify_systems(
        systems=(
            {"label": "graphene", "fdf": "materials/graphene/RUN.fdf", "basis_dir": "materials/graphene/basis"},
            {"label": "not_here", "fdf": "materials/does_not_exist/RUN.fdf", "basis_dir": "materials/graphene/basis"},
        ),
        checkpoint=None,
    )
    assert [record["label"] for record in report["systems"]] == ["graphene"]
    missing = next(item for item in report["cross_checks"] if item["check"] == "system_inputs_not_here")
    assert missing["outcome"] == "absent"
    # The remaining system agrees with itself; that must not certify the absent one.
    assert report["status"] == DECLARED_ONLY


@pytest.mark.slow
def test_checkpoint_basis_matches_the_ion_xml_contract():
    torch = pytest.importorskip("torch")
    del torch
    checkpoint = REPO_ROOT / (
        "Comparison/results/tbg_registry_spectral_loss/training/checkpoints/"
        "spectral-best-epoch=247-step=03472.ckpt"
    )
    if not checkpoint.is_file():
        pytest.skip("Graph2Mat checkpoint not present in this checkout")

    report = certify_systems(checkpoint=checkpoint)
    assert report["checkpoint"]["point_bases"][0]["basis"] == [[1, 0, 1], [1, 1, -1]]
    assert report["checkpoint"]["point_bases"][0]["basis_size"] == 4
    assert report["checkpoint"]["point_bases"][0]["basis_convention"] == "siesta_spherical"
    for record in report["systems"]:
        assert _outcome(record, "checkpoint_basis_C") == "pass"
        assert _outcome(record, "checkpoint_basis_convention_C") == "pass"
        assert _outcome(record, "checkpoint_matrix_components") == "pass"
    assert report["status"] != INVALID
