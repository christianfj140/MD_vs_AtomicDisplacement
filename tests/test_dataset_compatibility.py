"""Fase 6 (audit): physical dataset compatibility from real artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ml_vs_siesta.dataset_compatibility import (  # noqa: E402
    GHOST_INCOMPATIBLE,
    GHOST_NOT_APPLICABLE,
    GHOST_PROVEN_COMPATIBLE,
    GHOST_PROVEN_INACTIVE,
    GHOST_UNPROVEN,
    build_dataset_compatibility_report,
    ghost_compatibility,
    kpoint_spacing_per_axis,
    orb_indx_unit_cell_species,
    species_activity_report,
)

PRIMITIVE_FDF = """
NumberOfAtoms 2
NumberOfSpecies 2
%block ChemicalSpeciesLabel
  1   6  C
  2  -1  Ghost-H
%endblock ChemicalSpeciesLabel
LatticeConstant      1.46700 Ang
%block LatticeVectors
   1.5  -0.8660254038  0.0
   1.5   0.8660254038  0.0
   0.0   0.0          20.0
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
   0.333 0.333 0.0 1
   0.667 0.667 0.0 1
%endblock AtomicCoordinatesAndAtomicSpecies
%block kgrid_Monkhorst_Pack
  20 0 0 0.0
  0 20 0 0.0
  0 0 1 0.0
%endblock kgrid_Monkhorst_Pack
XC.functional GGA
XC.authors PBE
MeshCutoff 600.0 Ry
ElectronicTemperature 0.075 eV
DM.Tolerance 1.d-4
"""

SUPERCELL_FDF = """
NumberOfAtoms 50
NumberOfSpecies 1
%block ChemicalSpeciesLabel
  1   6  C
%endblock ChemicalSpeciesLabel
LatticeConstant      1.46700 Ang
%block LatticeVectors
   7.5  -4.330127019  0.0
   7.5   4.330127019  0.0
   0.0   0.0         20.0
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
   0.066 0.066 0.0 1
   0.133 0.133 0.0 1
%endblock AtomicCoordinatesAndAtomicSpecies
%block kgrid_Monkhorst_Pack
  4 0 0 0.0
  0 4 0 0.0
  0 0 1 0.0
%endblock kgrid_Monkhorst_Pack
XC.functional GGA
XC.authors PBE
MeshCutoff 600.0 Ry
ElectronicTemperature 0.075 eV
DM.Tolerance 1.d-4
"""

ORB_INDX_NO_GHOST = """      8     200 = orbitals in unit cell and supercell. See end of file.

    io    ia is   spec iao  n  l  m  z  p          sym      rc    isc     iuo
     1     1  1      C   1  2  0  0  1  F            s   4.089  0  0  0     1
     2     1  1      C   2  2  1 -1  1  F           py   4.868  0  0  0     2
     3     1  1      C   3  2  1  0  1  F           pz   4.868  0  0  0     3
     4     1  1      C   4  2  1  1  1  F           px   4.868  0  0  0     4
     5     2  1      C   1  2  0  0  1  F            s   4.089  0  0  0     5
     6     2  1      C   2  2  1 -1  1  F           py   4.868  0  0  0     6
     7     2  1      C   3  2  1  0  1  F           pz   4.868  0  0  0     7
     8     2  1      C   4  2  1  1  1  F           px   4.868  0  0  0     8
     9     3  1      C   1  2  0  0  1  F            s   4.089  1  0  0     1
"""


def _sample(tmp_path: Path, name: str, fdf: str, orb_indx: str | None = ORB_INDX_NO_GHOST):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "RUN.fdf").write_text(fdf, encoding="utf-8")
    if orb_indx is not None:
        (d / "graphene.ORB_INDX").write_text(orb_indx, encoding="utf-8")
    return d


PROV_WITH_GHOST = {
    "species": [
        {"index": 1, "atomic_number": 6, "label": "C"},
        {"index": 2, "atomic_number": -1, "label": "Ghost-H"},
    ],
    "basis_file_sha256": {"C.ion.xml": "aaa", "Ghost-H.ion.xml": "ggg"},
    "pseudopotential_sha256": {"C": "ppp"},
}
PROV_NO_GHOST = {
    "species": [{"index": 1, "atomic_number": 6, "label": "C"}],
    "basis_file_sha256": {"C.ion.xml": "aaa"},
    "pseudopotential_sha256": {"C": "ppp"},
}


def test_orb_indx_parse_counts_unit_cell_orbitals(tmp_path):
    path = tmp_path / "graphene.ORB_INDX"
    path.write_text(ORB_INDX_NO_GHOST, encoding="utf-8")
    assert orb_indx_unit_cell_species(path) == {"C": 8}
    assert orb_indx_unit_cell_species(tmp_path / "missing.ORB_INDX") is None


def test_kdensity_primitive_20x20_matches_supercell_4x4():
    small = kpoint_spacing_per_axis(PRIMITIVE_FDF)
    large = kpoint_spacing_per_axis(SUPERCELL_FDF)
    assert small is not None and large is not None
    for s, l in zip(small[:2], large[:2]):
        assert abs(s - l) / s < 0.01, (small, large)


def test_declared_only_ghost_is_proven_inactive(tmp_path):
    small_dir = _sample(tmp_path, "small", PRIMITIVE_FDF)
    large_dir = _sample(tmp_path, "large", SUPERCELL_FDF)
    report = build_dataset_compatibility_report(
        small_dir, large_dir, PROV_WITH_GHOST, PROV_NO_GHOST
    )
    assert report["ghost_compatibility_status"] == GHOST_PROVEN_INACTIVE
    assert report["compatible"] is True
    assert report["blocking_errors"] == []


def test_active_ghost_in_one_dataset_is_incompatible(tmp_path):
    fdf_active_ghost = PRIMITIVE_FDF.replace(
        "   0.667 0.667 0.0 1", "   0.667 0.667 0.0 1\n   0.5 0.5 0.0 2"
    )
    small_dir = _sample(tmp_path, "small", fdf_active_ghost)
    large_dir = _sample(tmp_path, "large", SUPERCELL_FDF)
    report = build_dataset_compatibility_report(
        small_dir, large_dir, PROV_WITH_GHOST, PROV_NO_GHOST
    )
    assert report["ghost_compatibility_status"] == GHOST_INCOMPATIBLE
    assert report["compatible"] is False


def test_ghost_without_evidence_is_unproven(tmp_path):
    fdf_no_coords = "\n".join(
        line for line in PRIMITIVE_FDF.splitlines()
        if "AtomicCoordinates" not in line and not line.strip().startswith("0.")
    )
    small_dir = _sample(tmp_path, "small", fdf_no_coords, orb_indx=None)
    large_dir = _sample(tmp_path, "large", SUPERCELL_FDF, orb_indx=None)
    status = build_dataset_compatibility_report(
        small_dir, large_dir, PROV_WITH_GHOST, PROV_NO_GHOST
    )["ghost_compatibility_status"]
    assert status == GHOST_UNPROVEN


def test_no_ghosts_anywhere_is_not_applicable(tmp_path):
    small_dir = _sample(tmp_path, "small", SUPERCELL_FDF)
    large_dir = _sample(tmp_path, "large", SUPERCELL_FDF)
    report = build_dataset_compatibility_report(
        small_dir, large_dir, PROV_NO_GHOST, PROV_NO_GHOST
    )
    assert report["ghost_compatibility_status"] == GHOST_NOT_APPLICABLE


def test_equal_ghost_sets_with_same_basis_are_proven_compatible():
    report_small = {"declared_ghost_species": ["Ghost-H"], "active_atomic_species": None,
                    "active_orbital_species": None}
    report_large = {"declared_ghost_species": ["Ghost-H"], "active_atomic_species": None,
                    "active_orbital_species": None}
    result = ghost_compatibility(
        report_small, report_large,
        small_basis={"Ghost-H.ion.xml": "g"}, large_basis={"Ghost-H.ion.xml": "g"},
    )
    assert result["status"] == GHOST_PROVEN_COMPATIBLE


def test_different_mesh_cutoff_blocks(tmp_path):
    small_dir = _sample(tmp_path, "small", PRIMITIVE_FDF)
    large_dir = _sample(
        tmp_path, "large", SUPERCELL_FDF.replace("MeshCutoff 600.0 Ry", "MeshCutoff 300.0 Ry")
    )
    report = build_dataset_compatibility_report(
        small_dir, large_dir, PROV_WITH_GHOST, PROV_NO_GHOST
    )
    assert report["compatible"] is False
    assert any("MeshCutoff" in error for error in report["blocking_errors"])


def test_different_kdensity_blocks(tmp_path):
    small_dir = _sample(tmp_path, "small", PRIMITIVE_FDF)
    large_dir = _sample(
        tmp_path, "large",
        SUPERCELL_FDF.replace("  4 0 0 0.0", "  8 0 0 0.0").replace("  0 4 0 0.0", "  0 8 0 0.0"),
    )
    report = build_dataset_compatibility_report(
        small_dir, large_dir, PROV_WITH_GHOST, PROV_NO_GHOST
    )
    assert report["compatible"] is False
    assert any("k-point density" in error for error in report["blocking_errors"])


def test_incompatible_pseudopotential_blocks(tmp_path):
    small_dir = _sample(tmp_path, "small", PRIMITIVE_FDF)
    large_dir = _sample(tmp_path, "large", SUPERCELL_FDF)
    other = dict(PROV_NO_GHOST, pseudopotential_sha256={"C": "DIFFERENT"})
    report = build_dataset_compatibility_report(
        small_dir, large_dir, PROV_WITH_GHOST, other
    )
    assert report["compatible"] is False
    assert any("pseudopotential" in error for error in report["blocking_errors"])


def test_real_repo_fdfs_are_kdensity_equivalent():
    primitive = (REPO_ROOT / "materials" / "graphene" / "RUN.fdf").read_text(encoding="utf-8")
    supercell = (REPO_ROOT / "materials" / "graphene_5x5" / "RUN.fdf").read_text(encoding="utf-8")
    small = kpoint_spacing_per_axis(primitive)
    large = kpoint_spacing_per_axis(supercell)
    if small is None or large is None:
        pytest.skip("repo fdfs not parseable")
    for s, l in zip(small[:2], large[:2]):
        assert abs(s - l) / s < 0.01
