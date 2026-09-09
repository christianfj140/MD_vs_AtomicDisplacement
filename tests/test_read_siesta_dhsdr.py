"""C06 / E-F_001-S8: the SIESTA ``dHSdR.nc`` reader.

No such file exists in this checkout (C02 inventory: nothing has produced one
yet), so every test writes a fixture with netCDF4 that reproduces the layout of
``Src/dhsdr_m.F90`` at the pinned runtime commit, *including* its two known
metadata defects: ``dH`` labelled ``unit = "Ry"`` when the numbers are Ry/Bohr,
and ``dS`` carrying no unit attribute at all. A reader that trusted either would
pass none of the unit tests below.

The ORB_INDX half runs on the repository's real graphene ORB_INDX (8 unit-cell
orbitals, 25 images), so the R-vector table and the orbital -> atom map are the
ones SIESTA actually printed, not invented ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

pytest.importorskip("netCDF4")

from orbital_contract import DECLARED_ONLY, INVALID, VERIFIED, parse_orb_indx  # noqa: E402
from read_siesta_dhsdr import (  # noqa: E402
    BOHR_TO_ANG,
    RY_TO_EV,
    DhsdrSchemaError,
    read_dhsdr,
)

# SIESTA orders the images with the origin cell first.
TINY_ISC = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int32)
TINY_NSC = (3, 1, 1)
TINY_NO_U = 4
TINY_NA_U = 2
TINY_ORBITAL_ATOM = np.array([1, 1, 2, 2])  # two orbitals per atom


# ---------------------------------------------------------------------------
# Fixture writer: the layout of Src/dhsdr_m.F90, defects included
# ---------------------------------------------------------------------------


def write_dhsdr(
    path,
    *,
    blocks,
    no_u=TINY_NO_U,
    n_spin=1,
    na_u=TINY_NA_U,
    nsc=TINY_NSC,
    isc_off=TINY_ISC,
    dx=0.04,
    dh_tol=-1.0,
    ds_tol=-1.0,
    dh_unit="Ry",
    ds_unit=None,
    atom_index=None,
    omit=(),
    extra_root_variable=False,
    xyz=3,
    transpose_dh=False,
):
    """Write one dHSdR.nc. Every argument exists so a test can break exactly one."""
    from netCDF4 import Dataset

    isc_off = np.asarray(isc_off, dtype=np.int32)
    if atom_index is None:
        atom_index = sorted({atom for atom, _ in blocks})

    with Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("no_u", no_u)
        dataset.createDimension("spin", n_spin)
        dataset.createDimension("na_u", na_u)
        dataset.createDimension("xyz", xyz)
        dataset.createDimension("one", 1)

        for name, value, unit in (
            ("dHdR.Tolerance", dh_tol, "Ry/Bohr"),
            ("dSdR.Tolerance", ds_tol, "1/Bohr"),
            ("FC.Displacement", dx, "Bohr"),
        ):
            if name in omit:
                continue
            variable = dataset.createVariable(name, "f8", ("one",))
            variable.info = name
            variable.unit = unit
            variable[:] = [value]

        if extra_root_variable:
            dataset.createVariable("isa", "i4", ("na_u",))[:] = np.ones(na_u, dtype=np.int32)

        displacements = dataset.createGroup("DISPLACEMENTS")
        displacements.createDimension("n_disp", len(atom_index))
        variable = displacements.createVariable("atom_index", "i4", ("n_disp",))
        variable.info = "Indices of displaced atoms"
        variable[:] = np.asarray(atom_index, dtype=np.int32)

        for (atom, axis), data in sorted(blocks.items()):
            atom_group = displacements.groups.get(str(atom)) or displacements.createGroup(str(atom))
            axis_group = atom_group.createGroup(str(axis))
            axis_group.createDimension("n_s", isc_off.shape[0])
            axis_group.createVariable("nsc", "i4", ("xyz",))[:] = np.asarray(nsc, dtype=np.int32)[:xyz]
            axis_group.createVariable("isc_off", "i4", ("n_s", "xyz"))[:] = isc_off[:, :xyz]

            for name, unit in (("dH", dh_unit), ("dS", ds_unit)):
                n_col, list_col, values = data[name]
                group = axis_group.createGroup(name)
                group.createDimension("nnzs", len(list_col))
                group.createVariable("n_col", "i4", ("no_u",))[:] = np.asarray(n_col, dtype=np.int32)
                group.createVariable("list_col", "i4", ("nnzs",))[:] = np.asarray(list_col, dtype=np.int32)
                if name == "dH":
                    dims = ("nnzs", "spin") if transpose_dh else ("spin", "nnzs")
                else:
                    dims = ("nnzs",)
                variable = group.createVariable(name, "f8", dims)
                variable.info = name
                if unit is not None:
                    variable.unit = unit
                variable[:] = np.asarray(values, dtype=np.float64)
    return path


def _rows(no_u, n_s, keep):
    """CSR rows over the whole supercell, keeping the columns ``keep`` allows."""
    n_col, list_col = [], []
    for row in range(no_u):
        columns = [j for j in range(1, no_u * n_s + 1) if keep(row, (j - 1) % no_u)]
        n_col.append(len(columns))
        list_col.extend(columns)
    return np.array(n_col), np.array(list_col)


def _blocks(rng, *, atoms=(1, 2), axes=(1, 2, 3), no_u=TINY_NO_U, n_s=3, n_spin=1, orbital_atom=TINY_ORBITAL_ATOM, ds_leak=False):
    """dH dense over the supercell, dS filtered onto the displaced atom."""
    blocks = {}
    for atom in atoms:
        for axis in axes:
            n_col_h, list_col_h = _rows(no_u, n_s, lambda row, col: True)
            values_h = rng.normal(size=(n_spin, len(list_col_h)))

            def keep(row, col, atom=atom):
                if ds_leak:
                    return True
                return orbital_atom[row] == atom or orbital_atom[col] == atom

            n_col_s, list_col_s = _rows(no_u, n_s, keep)
            values_s = rng.normal(size=len(list_col_s))
            blocks[(atom, axis)] = {
                "dH": (n_col_h, list_col_h, values_h),
                "dS": (n_col_s, list_col_s, values_s),
            }
    return blocks


@pytest.fixture()
def tiny(tmp_path):
    rng = np.random.default_rng(20260901)
    blocks = _blocks(rng)
    path = write_dhsdr(tmp_path / "graphene.dHSdR.nc", blocks=blocks)
    return path, blocks


def _outcome(checks, name):
    return next(item["outcome"] for item in checks if item["check"] == name)


# ---------------------------------------------------------------------------
# Atom / axis / spin / sparse indexing
# ---------------------------------------------------------------------------


def test_atom_axis_and_kind_are_addressable(tiny):
    path, blocks = tiny
    dhsdr = read_dhsdr(path)

    assert dhsdr.displaced_atoms == (1, 2)
    assert sorted(dhsdr.records) == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    record = dhsdr.derivative(2, 3, "D_H")
    assert (record.atom, record.axis, record.axis_label) == (2, 3, "z")
    assert record.nnz == len(blocks[(2, 3)]["dH"][1])
    assert dhsdr.derivative(2, 3, "D_S").nnz == len(blocks[(2, 3)]["dS"][1])

    with pytest.raises(KeyError):
        dhsdr.derivative(3, 1)
    with pytest.raises(KeyError):
        dhsdr.derivative(1, 1, "D_X")


def test_spin_dimension_is_read_from_the_file_not_assumed(tmp_path):
    rng = np.random.default_rng(7)
    blocks = _blocks(rng, atoms=(1,), axes=(1,), n_spin=2)
    path = write_dhsdr(tmp_path / "spin.dHSdR.nc", blocks=blocks, n_spin=2)
    dhsdr = read_dhsdr(path)

    d_h = dhsdr.derivative(1, 1, "D_H")
    assert dhsdr.n_spin == 2
    assert d_h.spin_resolved and d_h.n_spin == 2
    # Fortran stores dH as (nnzs, spin), i.e. (spin, nnzs) here; the reader
    # transposes it, so channel 1 must be the second written row.
    np.testing.assert_allclose(d_h.values_raw[:, 1], blocks[(1, 1)]["dH"][2][1])
    assert not np.allclose(d_h.values_raw[:, 0], d_h.values_raw[:, 1])

    d_s = dhsdr.derivative(1, 1, "D_S")
    assert d_s.spin_resolved is False and d_s.n_spin == 1


def test_sparse_indices_round_trip_through_the_supercell_rule(tiny):
    path, blocks = tiny
    record = read_dhsdr(path).derivative(1, 2, "D_H")
    n_col, list_col, values = blocks[(1, 2)]["dH"]

    np.testing.assert_array_equal(record.n_col, n_col)
    np.testing.assert_array_equal(record.supercell_column, list_col)
    np.testing.assert_array_equal(record.row, np.repeat(np.arange(len(n_col)), n_col))
    np.testing.assert_array_equal(record.col, (list_col - 1) % record.no_u)
    np.testing.assert_array_equal(record.image, (list_col - 1) // record.no_u)
    np.testing.assert_array_equal(record.isc, TINY_ISC[record.image])
    # Image 0 is the origin cell, and every image of the file is used.
    np.testing.assert_array_equal(record.isc_off[0], [0, 0, 0])
    assert sorted(set(record.image.tolist())) == [0, 1, 2]

    dense = record.to_dense()
    np.testing.assert_allclose(dense[record.image, record.row, record.col], record.values[:, 0])
    np.testing.assert_allclose(dense.sum(), record.values[:, 0].sum())
    np.testing.assert_allclose(dense[0, 0, 0], values[0, 0] * record.unit_factor)

    with pytest.raises(ValueError, match="dense form"):
        record.to_dense(max_elements=4)


# ---------------------------------------------------------------------------
# Units: never inferred, always converted with registered factors
# ---------------------------------------------------------------------------


def test_conversion_constants_match_an_independent_source():
    sisl_units = pytest.importorskip("sisl.unit.siesta")
    assert RY_TO_EV == pytest.approx(sisl_units.unit_convert("Ry", "eV"), rel=1e-12)
    assert BOHR_TO_ANG == pytest.approx(sisl_units.unit_convert("Bohr", "Ang"), rel=1e-12)


def test_values_and_thresholds_are_converted_to_canonical_units(tiny):
    path, blocks = tiny
    dhsdr = read_dhsdr(path)

    d_h = dhsdr.derivative(1, 1, "D_H")
    d_s = dhsdr.derivative(1, 1, "D_S")
    assert (d_h.unit_file, d_h.unit_canonical) == ("Ry/Bohr", "eV/Ang")
    assert (d_s.unit_file, d_s.unit_canonical) == ("1/Bohr", "1/Ang")

    # Hand-computed: Ry/Bohr -> eV/Ang is Ry->eV divided by Bohr->Ang.
    np.testing.assert_allclose(d_h.values, d_h.values_raw * (RY_TO_EV / BOHR_TO_ANG), rtol=0, atol=0)
    np.testing.assert_allclose(d_h.values_raw[:, 0], blocks[(1, 1)]["dH"][2][0])
    assert d_h.unit_factor == pytest.approx(25.711043, rel=1e-6)
    np.testing.assert_allclose(d_s.values, d_s.values_raw / BOHR_TO_ANG, rtol=1e-15, atol=0)
    assert d_s.unit_factor == pytest.approx(1.8897261, rel=1e-6)

    assert dhsdr.displacement_bohr == pytest.approx(0.04)
    assert dhsdr.displacement_ang == pytest.approx(0.04 * BOHR_TO_ANG)
    assert dhsdr.dhdr_tolerance_ry_bohr == pytest.approx(-1.0)
    assert dhsdr.dhdr_tolerance_ev_ang == pytest.approx(-RY_TO_EV / BOHR_TO_ANG)
    assert dhsdr.dsdr_tolerance_inv_ang == pytest.approx(-1.0 / BOHR_TO_ANG)


def test_the_wrong_unit_attribute_is_recorded_but_not_believed(tiny):
    path, _ = tiny
    provenance = read_dhsdr(path).derivative(1, 1, "D_H").unit_provenance

    # The file says "Ry" for a quantity that is Ry/Bohr. Both facts survive.
    assert provenance["attribute_in_file"] == "Ry"
    assert provenance["attribute_is_authoritative"] is False
    assert provenance["unit_of_stored_values"] == "Ry/Bohr"
    assert provenance["certified_numerically"] is False
    assert provenance["factor_constants"] == {"RY_TO_EV": RY_TO_EV, "BOHR_TO_ANG": BOHR_TO_ANG}
    assert _outcome(read_dhsdr(path).checks, "units_not_inferred") == "pass"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dh_unit": "eV"},
        {"dh_unit": None},  # a release that stops labelling dH is not this one
        {"ds_unit": "1/Ang"},
    ],
)
def test_unknown_unit_attributes_are_refused(tmp_path, kwargs):
    blocks = _blocks(np.random.default_rng(1), atoms=(1,), axes=(1,))
    path = write_dhsdr(tmp_path / "units.dHSdR.nc", blocks=blocks, **kwargs)
    with pytest.raises(DhsdrSchemaError, match="unit attribute"):
        read_dhsdr(path)


# ---------------------------------------------------------------------------
# Schema rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"omit": ("FC.Displacement",)}, "missing variables"),
        ({"omit": ("dSdR.Tolerance",)}, "missing variables"),
        ({"extra_root_variable": True}, "unrecognised variables"),
        ({"xyz": 2}, "dimension 'xyz' is 2"),
        ({"nsc": (2, 1, 1)}, "n_s = 3 but nsc"),
        ({"isc_off": np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]])}, "expected the origin cell"),
        ({"atom_index": [2]}, "not in atom_index"),
        ({"transpose_dh": True}, "dH has shape"),
    ],
)
def test_unrecognised_schema_is_refused(tmp_path, kwargs, match):
    blocks = _blocks(np.random.default_rng(2), atoms=(1,), axes=(1,))
    path = write_dhsdr(tmp_path / "broken.dHSdR.nc", blocks=blocks, **kwargs)
    with pytest.raises(DhsdrSchemaError, match=match):
        read_dhsdr(path)


def test_inconsistent_sparse_bookkeeping_is_refused(tmp_path):
    rng = np.random.default_rng(3)
    blocks = _blocks(rng, atoms=(1,), axes=(1,))
    n_col, list_col, values = blocks[(1, 1)]["dH"]
    blocks[(1, 1)]["dH"] = (n_col.copy(), list_col[:-1], values[:, :-1])
    path = write_dhsdr(tmp_path / "ncol.dHSdR.nc", blocks=blocks)
    with pytest.raises(DhsdrSchemaError, match="sum\\(n_col\\)"):
        read_dhsdr(path)

    blocks = _blocks(rng, atoms=(1,), axes=(1,))
    n_col, list_col, values = blocks[(1, 1)]["dH"]
    out_of_range = list_col.copy()
    out_of_range[-1] = TINY_NO_U * 3 + 1
    blocks[(1, 1)]["dH"] = (n_col, out_of_range, values)
    path = write_dhsdr(tmp_path / "range.dHSdR.nc", blocks=blocks)
    with pytest.raises(DhsdrSchemaError, match="outside the supercell range"):
        read_dhsdr(path)


def test_bad_axis_group_is_refused(tmp_path):
    blocks = _blocks(np.random.default_rng(4), atoms=(1,), axes=(4,))
    path = write_dhsdr(tmp_path / "axis.dHSdR.nc", blocks=blocks)
    with pytest.raises(DhsdrSchemaError, match="axis group"):
        read_dhsdr(path)


def test_non_strict_records_the_extra_content_instead_of_refusing(tmp_path):
    blocks = _blocks(np.random.default_rng(5), atoms=(1,), axes=(1, 2, 3))
    path = write_dhsdr(tmp_path / "extra.dHSdR.nc", blocks=blocks, extra_root_variable=True)
    dhsdr = read_dhsdr(path, strict=False)
    assert _outcome(dhsdr.checks, "unrecognised_variables_in_/") == "absent"
    # Unverified, never VERIFIED: an unknown variable is an unknown release.
    assert dhsdr.status == DECLARED_ONLY


def test_partial_axes_are_read_but_flagged(tmp_path):
    blocks = _blocks(np.random.default_rng(6), atoms=(1,), axes=(1, 2))
    path = write_dhsdr(tmp_path / "partial.dHSdR.nc", blocks=blocks)
    dhsdr = read_dhsdr(path)
    assert _outcome(dhsdr.checks, "all_three_axes_per_atom") == "absent"
    assert dhsdr.status == DECLARED_ONLY


# ---------------------------------------------------------------------------
# ORB_INDX: orbital -> atom map, R-vector table, dS filter
# ---------------------------------------------------------------------------


def _graphene_orb_indx():
    matches = sorted(REPO_ROOT.glob("Comparison/datasets/graphene_w90_snapshot_scaling/**/graphene.ORB_INDX"))
    if not matches:
        pytest.skip("no graphene ORB_INDX in this checkout")
    return matches[0]


def _graphene_fixture(tmp_path, *, ds_leak=False, no_u=None, isc_off=None):
    """A fixture whose supercell is the one the real graphene ORB_INDX prints."""
    orb = parse_orb_indx(_graphene_orb_indx())
    real_no_u = orb["orbitals_unit_cell"]
    n_s = orb["supercell_images"]
    table = np.array(
        [orb["rows"][image * real_no_u]["isc"] for image in range(n_s)], dtype=np.int32
    )
    orbital_atom = np.array([row["ia"] for row in orb["rows"][:real_no_u]])
    nsc = [int(1 + 2 * abs(table[:, axis]).max()) for axis in range(3)]

    blocks = _blocks(
        np.random.default_rng(11),
        atoms=(1, 2),
        no_u=real_no_u,
        n_s=n_s,
        orbital_atom=orbital_atom,
        ds_leak=ds_leak,
    )
    path = write_dhsdr(
        tmp_path / "graphene.dHSdR.nc",
        blocks=blocks,
        no_u=real_no_u if no_u is None else no_u,
        na_u=int(orbital_atom.max()),
        nsc=nsc,
        isc_off=table if isc_off is None else isc_off,
    )
    return path, orb


def test_orb_indx_certifies_the_r_vector_table_and_the_ds_filter(tmp_path):
    path, orb = _graphene_fixture(tmp_path)
    dhsdr = read_dhsdr(path, orb_indx=_graphene_orb_indx())

    assert dhsdr.no_u == orb["orbitals_unit_cell"]
    assert _outcome(dhsdr.checks, "orb_indx_no_u") == "pass"
    assert _outcome(dhsdr.checks, "r_vector_table_matches_orb_indx") == "pass"
    assert _outcome(dhsdr.checks, "supercell_image_count") == "pass"
    assert _outcome(dhsdr.checks, "ds_restricted_to_displaced_atom") == "pass"
    assert dhsdr.status == VERIFIED


def test_without_orb_indx_the_mapping_stays_unverified(tmp_path):
    path, _ = _graphene_fixture(tmp_path)
    dhsdr = read_dhsdr(path)
    assert _outcome(dhsdr.checks, "orb_indx_cross_check") == "absent"
    assert dhsdr.status == DECLARED_ONLY


def test_ds_elements_off_the_displaced_atom_fail_the_filter_check(tmp_path):
    path, _ = _graphene_fixture(tmp_path, ds_leak=True)
    dhsdr = read_dhsdr(path, orb_indx=_graphene_orb_indx())
    assert _outcome(dhsdr.checks, "ds_restricted_to_displaced_atom") == "fail"
    assert dhsdr.status == INVALID


def test_a_shifted_r_vector_table_is_caught(tmp_path):
    orb = parse_orb_indx(_graphene_orb_indx())
    n_s = orb["supercell_images"]
    no_u = orb["orbitals_unit_cell"]
    table = np.array([orb["rows"][image * no_u]["isc"] for image in range(n_s)], dtype=np.int32)
    shifted = table.copy()
    shifted[1:] = shifted[1:][::-1]  # same set of images, wrong order
    path, _ = _graphene_fixture(tmp_path, isc_off=shifted)
    dhsdr = read_dhsdr(path, orb_indx=_graphene_orb_indx())
    assert _outcome(dhsdr.checks, "r_vector_table_matches_orb_indx") == "fail"
    assert dhsdr.status == INVALID


def test_a_different_orbital_count_is_caught(tmp_path):
    path, _ = _graphene_fixture(tmp_path)
    other = sorted(REPO_ROOT.glob("Comparison/datasets/bilayer_graphene_AB_md30/**/bilayer_graphene_AB.ORB_INDX"))
    if not other:
        pytest.skip("no second ORB_INDX in this checkout")
    dhsdr = read_dhsdr(path, orb_indx=other[0])
    assert _outcome(dhsdr.checks, "orb_indx_no_u") == "fail"
    assert dhsdr.status == INVALID


# ---------------------------------------------------------------------------
# Metadata a downstream artifact has to carry
# ---------------------------------------------------------------------------


def test_metadata_carries_displacement_thresholds_units_and_provenance(tiny):
    path, _ = tiny
    metadata = read_dhsdr(path).metadata()

    assert metadata["derivative_method"] == "siesta_fc_dHS"
    assert metadata["sha256"] and len(metadata["sha256"]) == 64
    assert metadata["displacement"]["ang"] == pytest.approx(0.04 * BOHR_TO_ANG)
    assert metadata["thresholds"]["dHdR"]["effective"] is False
    assert metadata["thresholds"]["dSdR"]["effective"] is True
    assert metadata["units"]["D_H"]["canonical"] == "eV/Ang"
    assert metadata["units"]["D_S"]["canonical"] == "1/Ang"
    assert metadata["records"]["1/1"] == ["D_H", "D_S"]
    assert "central difference" in metadata["sign_convention"]
