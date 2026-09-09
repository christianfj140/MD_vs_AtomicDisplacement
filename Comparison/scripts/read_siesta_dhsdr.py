#!/usr/bin/env python3
"""C06 / E-F_001-S8: SIESTA ``*.dHSdR.nc`` -> canonical sparse ``D_H`` / ``D_S``.

``FC.Save.dHS`` makes SIESTA write, for every displaced atom and axis of an FC
run, the central difference

    dH/dR = (H(+dx) - H(-dx)) / (2 dx),      dS/dR likewise,

in the *same* supercell sparse format as the rest of SIESTA's NetCDF output.
This module turns that file into the objects the EPC formalism consumes, and
refuses to guess anything the file does not state.

Layout, read off ``Src/dhsdr_m.F90`` at ``4e9a460606aa`` (siesta
``5.4.2-11-g4e9a46060``, the runtime C01 pinned; see ``shared/run_inventory.py``
``SOURCE_CANDIDATE_EXPECTATIONS``). Fortran declares dimensions fastest-first,
so every NetCDF/python shape below is the reverse of the one in the source::

    /                       dims no_u, spin, na_u, xyz=3, one=1
      dHdR.Tolerance(one)   unit "Ry/Bohr"
      dSdR.Tolerance(one)   unit "1/Bohr"
      FC.Displacement(one)  unit "Bohr"   <- dx, the half-step of the difference
      /DISPLACEMENTS        dim n_disp; atom_index(n_disp) = the displaced atoms
        /<ia>/<ixyz>        dim n_s; nsc(xyz); isc_off(n_s, xyz)
          /dH               dim nnzs; n_col(no_u), list_col(nnzs), dH(spin, nnzs)
          /dS               dim nnzs; n_col(no_u), list_col(nnzs), dS(nnzs)

Three things this reader will not do:

*No unit inference from names or attributes.* The 5.4.2 release labels ``dH``
with ``unit = "Ry"`` although the numbers are Ry/Bohr, and gives ``dS`` no unit
attribute at all. Attribute strings are therefore looked up in an explicit table
(:data:`DH_UNIT_ATTRIBUTES` / :data:`DS_UNIT_ATTRIBUTES`) and an unlisted string
is an error, never a parse. The applied factor and its two constants travel with
every record in ``unit_provenance``; GO-2 still has to confirm them numerically
against explicit finite differences (C09).

*No re-implementation of the orbital/R-vector mapping.* ``dHSdR.nc`` carries no
geometry, no species and no ``iaorb``: which atom a row belongs to comes from
ORB_INDX through :func:`orbital_contract.parse_orb_indx`, and the image
bookkeeping is the file's own ``isc_off``, cross-checked against the ORB_INDX
``isc`` column.

*No silent acceptance of an unknown file.* Dimensions, variables, groups,
shapes, ``sum(n_col) == nnzs`` and the column range are all checked, and a file
that is not this schema raises :class:`DhsdrSchemaError` instead of being
decoded into plausible-looking numbers.

Two properties of this release that the checks record rather than hide:

* ``FC.dHdR.Tolerance`` is a no-op in ``calc_dHdR`` (``discard`` starts
  ``.false.`` and is accumulated with ``.and.``), so ``dH`` comes out dense;
* ``dS`` carries a real filter: an element survives only if the row *or* the
  column orbital belongs to the displaced atom. With an ORB_INDX at hand that
  becomes the ``ds_restricted_to_displaced_atom`` check, and it is exactly why
  the memo (section 6) can split inter-atomic ``S^L``/``S^R`` by displaced atom.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
_SHARED_DIR = REPO_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from benchmark_manifest import file_sha256  # noqa: E402
from fdf_materialization import BOHR_TO_ANG  # noqa: E402
from orbital_contract import (  # noqa: E402
    DECLARED_ONLY,
    INVALID,
    VERIFIED,
    parse_orb_indx,
)

SCHEMA = "siesta_dhsdr_v1"
SOURCE_REFERENCE = "Src/dhsdr_m.F90 @ 4e9a460606aa14320e56cb0edd88e19d717a46b2 (siesta 5.4.2-11-g4e9a46060)"

# CODATA 2018. BOHR_TO_ANG is the repository's single copy (shared/fdf_materialization.py).
RY_TO_EV = 13.605693122994

# What the file may say, and what it means. An unlisted string is refused.
DH_UNIT_ATTRIBUTES = {
    # 5.4.2 writes "Ry"; the value is a derivative and is in Ry/Bohr. The label
    # is wrong in the file, so it is not trusted, only recognised.
    "Ry": "Ry/Bohr",
    "Ry/Bohr": "Ry/Bohr",
}
DS_UNIT_ATTRIBUTES = {
    # 5.4.2 writes no unit attribute for dS at all.
    None: "1/Bohr",
    "1/Bohr": "1/Bohr",
}
DISPLACEMENT_UNIT_ATTRIBUTES = {"Bohr": "Bohr"}
DH_TOLERANCE_UNIT_ATTRIBUTES = {"Ry/Bohr": "Ry/Bohr"}
DS_TOLERANCE_UNIT_ATTRIBUTES = {"1/Bohr": "1/Bohr"}

# Canonical units of this repository (see epc_formalism.UNITS).
UNIT_FACTORS = {
    "Ry/Bohr": (RY_TO_EV / BOHR_TO_ANG, "eV/Ang"),
    "1/Bohr": (1.0 / BOHR_TO_ANG, "1/Ang"),
    "Bohr": (BOHR_TO_ANG, "Ang"),
}

ROOT_DIMENSIONS = {"no_u": None, "spin": None, "na_u": None, "xyz": 3, "one": 1}
ROOT_VARIABLES = ("dHdR.Tolerance", "dSdR.Tolerance", "FC.Displacement")
DISPLACEMENT_GROUP = "DISPLACEMENTS"
AXES = {1: "x", 2: "y", 3: "z"}
KINDS = {"D_H": "dH", "D_S": "dS"}

SIGN_CONVENTION = (
    "central difference (H(+dx) - H(-dx)) / (2 dx) with respect to the cartesian "
    "position of the displaced atom; the H_p/H_m inversion of 5.4.0 was fixed in "
    "5cfda23a, which is an ancestor of the pinned runtime"
)
SUPERCELL_RULE = "list_col j (1-based): col = (j - 1) mod no_u, image = (j - 1) // no_u, R = isc_off[image] @ cell"


class DhsdrSchemaError(ValueError):
    """The file is not the ``dHSdR.nc`` schema this reader certifies."""


def _check(name: str, outcome: str, detail: str = "") -> dict[str, str]:
    """One recorded check. ``outcome`` is pass | fail | absent."""
    return {"check": name, "outcome": outcome, "detail": detail}


def _status(checks: list[dict[str, str]]) -> str:
    if any(item["outcome"] == "fail" for item in checks):
        return INVALID
    if any(item["outcome"] == "absent" for item in checks):
        return DECLARED_ONLY
    return VERIFIED


# ---------------------------------------------------------------------------
# One (atom, axis, kind) derivative
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseDerivative:
    """``dH/dR`` or ``dS/dR`` for one displaced atom and one cartesian axis.

    Rows are unit-cell orbitals, columns are unit-cell orbitals *plus* the
    periodic image they live in, exactly as SIESTA stores them::

        list_col[k] - 1 == col[k] + image[k] * no_u

    All index arrays are 0-based; ``supercell_column`` returns the file's own
    1-based column back, which is what the round-trip test asserts.
    """

    kind: str  # D_H | D_S
    atom: int  # 1-based SIESTA index of the displaced atom
    axis: int  # 1 | 2 | 3
    no_u: int
    n_spin: int
    spin_resolved: bool
    n_col: np.ndarray  # (no_u,) non-zeros per row
    row: np.ndarray  # (nnz,) unit-cell orbital, 0-based
    col: np.ndarray  # (nnz,) unit-cell orbital, 0-based
    image: np.ndarray  # (nnz,) index into isc_off, 0-based
    isc: np.ndarray  # (nnz, 3) lattice-vector offset of the column
    values_raw: np.ndarray  # (nnz, n_spin) as stored in the file
    unit_attribute: str | None  # what the file said, verbatim
    unit_file: str  # what that attribute means
    unit_canonical: str
    unit_factor: float
    nsc: np.ndarray  # (3,)
    isc_off: np.ndarray  # (n_s, 3)
    columns_ascending_per_row: bool

    @property
    def axis_label(self) -> str:
        return AXES[self.axis]

    @property
    def nnz(self) -> int:
        return int(self.values_raw.shape[0])

    @property
    def n_s(self) -> int:
        return int(self.isc_off.shape[0])

    @property
    def values(self) -> np.ndarray:
        """``values_raw`` in canonical units (eV/Ang for D_H, 1/Ang for D_S)."""
        return self.values_raw * self.unit_factor

    @property
    def supercell_column(self) -> np.ndarray:
        """The file's 1-based ``list_col``, rebuilt from (col, image)."""
        return self.col + self.image * self.no_u + 1

    @property
    def unit_provenance(self) -> dict[str, Any]:
        return {
            "attribute_in_file": self.unit_attribute,
            "attribute_is_authoritative": False,
            "unit_of_stored_values": self.unit_file,
            "unit_canonical": self.unit_canonical,
            "factor": self.unit_factor,
            "factor_constants": {"RY_TO_EV": RY_TO_EV, "BOHR_TO_ANG": BOHR_TO_ANG},
            "source": SOURCE_REFERENCE,
            "certified_numerically": False,  # GO-2 / C09 does that, not this reader
        }

    def to_dense(self, spin: int = 0, *, max_elements: int = 20_000_000) -> np.ndarray:
        """``(n_s, no_u, no_u)`` in canonical units. Small systems only."""
        size = self.n_s * self.no_u * self.no_u
        if size > max_elements:
            raise ValueError(
                f"dense form would be {size} elements ({self.n_s} images x {self.no_u}^2); "
                "keep the sparse arrays or raise max_elements deliberately"
            )
        dense = np.zeros((self.n_s, self.no_u, self.no_u), dtype=np.float64)
        dense[self.image, self.row, self.col] = self.values[:, spin]
        return dense

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "atom": self.atom,
            "axis": self.axis,
            "axis_label": self.axis_label,
            "nnz": self.nnz,
            "no_u": self.no_u,
            "n_spin": self.n_spin,
            "spin_resolved": self.spin_resolved,
            "n_s": self.n_s,
            "nsc": self.nsc.tolist(),
            "images_used": int(np.unique(self.image).size),
            "columns_ascending_per_row": self.columns_ascending_per_row,
            "max_abs_canonical": float(np.abs(self.values).max()) if self.nnz else 0.0,
            "units": self.unit_provenance,
        }


@dataclass(frozen=True)
class DhsdrFile:
    """Everything one ``*.dHSdR.nc`` states, decoded and checked."""

    schema: str
    path: str
    sha256: str | None
    no_u: int
    n_spin: int
    na_u: int
    displaced_atoms: tuple[int, ...]
    displacement_bohr: float
    displacement_ang: float
    dhdr_tolerance_ry_bohr: float
    dhdr_tolerance_ev_ang: float
    dsdr_tolerance_inv_bohr: float
    dsdr_tolerance_inv_ang: float
    records: dict[tuple[int, int], dict[str, SparseDerivative]]
    checks: list[dict[str, str]]

    @property
    def status(self) -> str:
        return _status(self.checks)

    def derivative(self, atom: int, axis: int, kind: str = "D_H") -> SparseDerivative:
        if kind not in KINDS:
            raise KeyError(f"kind must be one of {sorted(KINDS)}, got {kind!r}")
        try:
            return self.records[(int(atom), int(axis))][kind]
        except KeyError:
            raise KeyError(
                f"{kind} for atom {atom} axis {axis} is not in {self.path}; "
                f"it holds {sorted(self.records)}"
            ) from None

    def metadata(self) -> dict[str, Any]:
        """The provenance a downstream artifact has to carry (no matrices)."""
        return {
            "schema": self.schema,
            "path": self.path,
            "sha256": self.sha256,
            "source_reference": SOURCE_REFERENCE,
            "derivative_backend": "siesta",
            "derivative_method": "siesta_fc_dHS",
            "sign_convention": SIGN_CONVENTION,
            "supercell_rule": SUPERCELL_RULE,
            "no_u": self.no_u,
            "n_spin": self.n_spin,
            "na_u": self.na_u,
            "displaced_atoms": list(self.displaced_atoms),
            "displacement": {"bohr": self.displacement_bohr, "ang": self.displacement_ang},
            "thresholds": {
                "dHdR": {
                    "value_ry_bohr": self.dhdr_tolerance_ry_bohr,
                    "value_ev_ang": self.dhdr_tolerance_ev_ang,
                    "effective": False,
                    "note": "FC.dHdR.Tolerance is a no-op in calc_dHdR (discard "
                    "starts .false. and is accumulated with .and.); dH is dense",
                },
                "dSdR": {
                    "value_inv_bohr": self.dsdr_tolerance_inv_bohr,
                    "value_inv_ang": self.dsdr_tolerance_inv_ang,
                    "effective": True,
                    "note": "applied together with the displaced-atom filter: an "
                    "element survives only if row or column orbital sits on the "
                    "displaced atom and |dS| >= tolerance * 2 dx",
                },
            },
            "units": {
                "D_H": {"file": "Ry/Bohr", "canonical": "eV/Ang", "factor": UNIT_FACTORS["Ry/Bohr"][0]},
                "D_S": {"file": "1/Bohr", "canonical": "1/Ang", "factor": UNIT_FACTORS["1/Bohr"][0]},
                "constants": {"RY_TO_EV": RY_TO_EV, "BOHR_TO_ANG": BOHR_TO_ANG},
            },
            "records": {f"{atom}/{axis}": sorted(kinds) for (atom, axis), kinds in sorted(self.records.items())},
            "status": self.status,
            "checks": self.checks,
        }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _fail(where: str, message: str) -> None:
    raise DhsdrSchemaError(f"{where}: {message}")


def _dim(node: Any, name: str, where: str) -> int:
    if name not in node.dimensions:
        _fail(where, f"missing dimension {name!r}")
    return len(node.dimensions[name])


def _var(node: Any, name: str, where: str) -> Any:
    if name not in node.variables:
        _fail(where, f"missing variable {name!r}")
    return node.variables[name]


def _attribute(variable: Any, name: str) -> str | None:
    return variable.getncattr(name) if name in variable.ncattrs() else None


def _unit(variable: Any, table: dict[str | None, str], where: str) -> tuple[str | None, str, float, str]:
    """Look the file's unit attribute up in ``table``. Never parse it."""
    attribute = _attribute(variable, "unit")
    if attribute not in table:
        _fail(
            where,
            f"unit attribute {attribute!r} is not one this reader knows "
            f"({sorted(str(key) for key in table)}); refusing to infer a unit",
        )
    unit_file = table[attribute]
    factor, canonical = UNIT_FACTORS[unit_file]
    return attribute, unit_file, factor, canonical


def _exact_content(
    node: Any,
    where: str,
    *,
    dimensions: set[str],
    variables: set[str],
    groups: set[str] | None,
    strict: bool,
    checks: list[dict[str, str]],
) -> None:
    """Refuse (or, if not strict, record) anything this schema does not declare.

    ``groups=None`` means "the subgroups are data" (the ``<atom>``/``<axis>``
    names), so only the dimensions and variables are constrained.
    """
    contents = [
        ("dimensions", set(node.dimensions), dimensions),
        ("variables", set(node.variables), variables),
    ]
    if groups is not None:
        contents.append(("groups", set(node.groups), groups))
    for label, observed, expected in contents:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            _fail(where, f"missing {label} {missing}")
        if extra:
            if strict:
                _fail(where, f"unrecognised {label} {extra}; this is not {SCHEMA}")
            checks.append(
                _check(f"unrecognised_{label}_in_{where}", "absent", f"ignored {extra} (strict=False)")
            )


def _read_sparse(
    group: Any,
    *,
    kind: str,
    atom: int,
    axis: int,
    no_u: int,
    n_spin: int,
    nsc: np.ndarray,
    isc_off: np.ndarray,
    strict: bool,
    checks: list[dict[str, str]],
) -> SparseDerivative:
    variable_name = KINDS[kind]
    where = f"/DISPLACEMENTS/{atom}/{axis}/{variable_name}"
    _exact_content(
        group,
        where,
        dimensions={"nnzs"},
        variables={"n_col", "list_col", variable_name},
        groups=set(),
        strict=strict,
        checks=checks,
    )

    nnz = _dim(group, "nnzs", where)
    n_col = np.asarray(_var(group, "n_col", where)[:], dtype=np.int64)
    list_col = np.asarray(_var(group, "list_col", where)[:], dtype=np.int64)
    data = _var(group, variable_name, where)
    raw = np.asarray(data[:], dtype=np.float64)

    if n_col.shape != (no_u,):
        _fail(where, f"n_col has shape {n_col.shape}, expected ({no_u},)")
    if list_col.shape != (nnz,):
        _fail(where, f"list_col has shape {list_col.shape}, expected ({nnz},)")
    if int(n_col.sum()) != nnz:
        _fail(where, f"sum(n_col) = {int(n_col.sum())} but nnzs = {nnz}")

    # dH is declared (/'nnzs','spin'/) in Fortran, i.e. (spin, nnzs) here; dS
    # has no spin dimension at all.
    if kind == "D_H":
        if raw.shape != (n_spin, nnz):
            _fail(where, f"dH has shape {raw.shape}, expected ({n_spin}, {nnz})")
        values_raw = np.ascontiguousarray(raw.T)
        unit_table = DH_UNIT_ATTRIBUTES
        spin_resolved = True
    else:
        if raw.shape != (nnz,):
            _fail(where, f"dS has shape {raw.shape}, expected ({nnz},)")
        values_raw = raw.reshape(nnz, 1)
        unit_table = DS_UNIT_ATTRIBUTES
        spin_resolved = False

    attribute, unit_file, factor, canonical = _unit(data, unit_table, where)

    n_s = isc_off.shape[0]
    if nnz and (list_col.min() < 1 or list_col.max() > no_u * n_s):
        _fail(
            where,
            f"list_col spans [{int(list_col.min())}, {int(list_col.max())}], outside the "
            f"supercell range [1, {no_u * n_s}]",
        )

    row = np.repeat(np.arange(no_u, dtype=np.int64), n_col)
    image = (list_col - 1) // no_u
    col = (list_col - 1) % no_u

    ascending = True
    start = 0
    for count in n_col:
        stop = start + int(count)
        if count > 1 and not np.all(np.diff(list_col[start:stop]) > 0):
            ascending = False
            break
        start = stop

    return SparseDerivative(
        kind=kind,
        atom=atom,
        axis=axis,
        no_u=no_u,
        n_spin=values_raw.shape[1],
        spin_resolved=spin_resolved,
        n_col=n_col,
        row=row,
        col=col,
        image=image,
        isc=isc_off[image],
        values_raw=values_raw,
        unit_attribute=attribute,
        unit_file=unit_file,
        unit_canonical=canonical,
        unit_factor=factor,
        nsc=nsc,
        isc_off=isc_off,
        columns_ascending_per_row=ascending,
    )


def read_dhsdr(
    path: str | Path,
    *,
    orb_indx: str | Path | None = None,
    strict: bool = True,
) -> DhsdrFile:
    """Decode one ``*.dHSdR.nc``. Raises :class:`DhsdrSchemaError` if it is not one."""
    from netCDF4 import Dataset  # lazy: only this reader needs netCDF4

    path = Path(path)
    checks: list[dict[str, str]] = []
    records: dict[tuple[int, int], dict[str, SparseDerivative]] = {}

    with Dataset(str(path), "r") as dataset:
        dataset.set_auto_mask(False)
        _exact_content(
            dataset,
            "/",
            dimensions=set(ROOT_DIMENSIONS),
            variables=set(ROOT_VARIABLES),
            groups={DISPLACEMENT_GROUP},
            strict=strict,
            checks=checks,
        )
        sizes = {name: _dim(dataset, name, "/") for name in ROOT_DIMENSIONS}
        for name, expected in ROOT_DIMENSIONS.items():
            if expected is not None and sizes[name] != expected:
                _fail("/", f"dimension {name!r} is {sizes[name]}, expected {expected}")
        no_u, n_spin, na_u = sizes["no_u"], sizes["spin"], sizes["na_u"]

        # Every unit attribute is looked up, including the ones whose value is
        # only metadata: an unexpected string means an unknown release.
        _unit(_var(dataset, "FC.Displacement", "/"), DISPLACEMENT_UNIT_ATTRIBUTES, "/FC.Displacement")
        displacement_bohr = _scalar(dataset, "FC.Displacement")
        dh_tolerance = _scalar(dataset, "dHdR.Tolerance")
        ds_tolerance = _scalar(dataset, "dSdR.Tolerance")
        _unit(_var(dataset, "dHdR.Tolerance", "/"), DH_TOLERANCE_UNIT_ATTRIBUTES, "/dHdR.Tolerance")
        _unit(_var(dataset, "dSdR.Tolerance", "/"), DS_TOLERANCE_UNIT_ATTRIBUTES, "/dSdR.Tolerance")

        displacements = dataset.groups[DISPLACEMENT_GROUP]
        _exact_content(
            displacements,
            f"/{DISPLACEMENT_GROUP}",
            dimensions={"n_disp"},
            variables={"atom_index"},
            groups=None,
            strict=strict,
            checks=checks,
        )
        n_disp = _dim(displacements, "n_disp", f"/{DISPLACEMENT_GROUP}")
        declared_atoms = np.asarray(
            _var(displacements, "atom_index", f"/{DISPLACEMENT_GROUP}")[:], dtype=np.int64
        )
        if declared_atoms.shape != (n_disp,):
            _fail(f"/{DISPLACEMENT_GROUP}", f"atom_index has shape {declared_atoms.shape}, expected ({n_disp},)")
        if declared_atoms.size and (declared_atoms.min() < 1 or declared_atoms.max() > na_u):
            _fail(
                f"/{DISPLACEMENT_GROUP}",
                f"atom_index spans [{int(declared_atoms.min())}, {int(declared_atoms.max())}], "
                f"outside [1, na_u = {na_u}]",
            )

        for atom_key, atom_group in displacements.groups.items():
            where = f"/{DISPLACEMENT_GROUP}/{atom_key}"
            if not atom_key.isdigit():
                _fail(where, "group name is not an atom index")
            atom = int(atom_key)
            if atom not in set(declared_atoms.tolist()):
                _fail(where, f"atom {atom} is not in atom_index {declared_atoms.tolist()}")
            _exact_content(
                atom_group, where, dimensions=set(), variables=set(), groups=None, strict=strict, checks=checks
            )

            for axis_key, axis_group in atom_group.groups.items():
                where = f"/{DISPLACEMENT_GROUP}/{atom_key}/{axis_key}"
                if not axis_key.isdigit() or int(axis_key) not in AXES:
                    _fail(where, f"axis group {axis_key!r} is not one of {sorted(AXES)}")
                axis = int(axis_key)
                _exact_content(
                    axis_group,
                    where,
                    dimensions={"n_s"},
                    variables={"nsc", "isc_off"},
                    groups=set(KINDS.values()),
                    strict=strict,
                    checks=checks,
                )
                n_s = _dim(axis_group, "n_s", where)
                nsc = np.asarray(_var(axis_group, "nsc", where)[:], dtype=np.int64)
                isc_off = np.asarray(_var(axis_group, "isc_off", where)[:], dtype=np.int64)
                if nsc.shape != (3,):
                    _fail(where, f"nsc has shape {nsc.shape}, expected (3,)")
                if isc_off.shape != (n_s, 3):
                    _fail(where, f"isc_off has shape {isc_off.shape}, expected ({n_s}, 3)")
                if int(prod(nsc.tolist())) != n_s:
                    _fail(where, f"n_s = {n_s} but nsc = {nsc.tolist()} implies {int(prod(nsc.tolist()))}")
                if np.any(isc_off[0] != 0):
                    _fail(where, f"isc_off[0] = {isc_off[0].tolist()}, expected the origin cell")

                records[(atom, axis)] = {
                    kind: _read_sparse(
                        axis_group.groups[variable_name],
                        kind=kind,
                        atom=atom,
                        axis=axis,
                        no_u=no_u,
                        n_spin=n_spin,
                        nsc=nsc,
                        isc_off=isc_off,
                        strict=strict,
                        checks=checks,
                    )
                    for kind, variable_name in KINDS.items()
                }

    if not records:
        _fail(f"/{DISPLACEMENT_GROUP}", "no /<atom>/<axis> displacement group in the file")

    checks.append(
        _check(
            "displacement_groups",
            "pass",
            f"{len(records)} (atom, axis) pairs over atoms {sorted({atom for atom, _ in records})}; "
            f"declared atom_index = {declared_atoms.tolist()}",
        )
    )
    incomplete = [
        atom
        for atom in sorted({atom for atom, _ in records})
        if sorted({axis for other, axis in records if other == atom}) != [1, 2, 3]
    ]
    checks.append(
        _check(
            "all_three_axes_per_atom",
            "pass" if not incomplete else "absent",
            "every displaced atom carries axes 1, 2, 3"
            if not incomplete
            else f"atoms {incomplete} carry fewer than three axes (interrupted FC run?)",
        )
    )
    checks.append(
        _check(
            "sparse_ordering",
            "pass"
            if all(record.columns_ascending_per_row for kinds in records.values() for record in kinds.values())
            else "absent",
            "rows in unit-cell orbital order, supercell columns ascending within a row",
        )
    )
    checks.append(
        _check(
            "dh_tolerance_is_a_noop",
            "pass",
            f"dHdR.Tolerance = {dh_tolerance} Ry/Bohr is written to the file but never applied by "
            "calc_dHdR in the pinned release; dH is dense",
        )
    )
    checks.append(
        _check(
            "units_not_inferred",
            "pass",
            "dH attribute -> Ry/Bohr, dS attribute -> 1/Bohr through an explicit table; "
            "the numerical certification of both is GO-2 (C09), not this reader",
        )
    )

    dhsdr = DhsdrFile(
        schema=SCHEMA,
        path=str(path),
        sha256=file_sha256(path),
        no_u=no_u,
        n_spin=n_spin,
        na_u=na_u,
        displaced_atoms=tuple(int(value) for value in declared_atoms),
        displacement_bohr=displacement_bohr,
        displacement_ang=displacement_bohr * UNIT_FACTORS["Bohr"][0],
        dhdr_tolerance_ry_bohr=dh_tolerance,
        dhdr_tolerance_ev_ang=dh_tolerance * UNIT_FACTORS["Ry/Bohr"][0],
        dsdr_tolerance_inv_bohr=ds_tolerance,
        dsdr_tolerance_inv_ang=ds_tolerance * UNIT_FACTORS["1/Bohr"][0],
        records=records,
        checks=checks,
    )

    if orb_indx is None:
        checks.append(
            _check(
                "orb_indx_cross_check",
                "absent",
                "no ORB_INDX supplied: the orbital -> atom map and the R-vector table are "
                "unverified, so the dS displaced-atom filter cannot be checked either",
            )
        )
    else:
        checks.extend(cross_check_orb_indx(dhsdr, orb_indx))
    return dhsdr


def _scalar(dataset: Any, name: str) -> float:
    values = np.asarray(dataset.variables[name][:], dtype=np.float64).reshape(-1)
    if values.size != 1:
        _fail("/", f"{name} holds {values.size} values, expected 1")
    return float(values[0])


# ---------------------------------------------------------------------------
# ORB_INDX: the orbital -> atom map and the R-vector table
# ---------------------------------------------------------------------------


def cross_check_orb_indx(dhsdr: DhsdrFile, orb_indx: str | Path) -> list[dict[str, str]]:
    """Check the file against the ORB_INDX of the same run (C03's parser)."""
    orb = parse_orb_indx(orb_indx)
    checks: list[dict[str, str]] = []

    if orb["orbitals_unit_cell"] != dhsdr.no_u:
        return [
            _check(
                "orb_indx_no_u",
                "fail",
                f"ORB_INDX declares {orb['orbitals_unit_cell']} unit-cell orbitals, "
                f"the .nc declares no_u = {dhsdr.no_u}",
            )
        ]
    checks.append(_check("orb_indx_no_u", "pass", f"{dhsdr.no_u} unit-cell orbitals"))

    orbital_atom = np.zeros(dhsdr.no_u, dtype=np.int64)
    for row in orb["rows"]:
        if row["io"] <= dhsdr.no_u:
            orbital_atom[row["io"] - 1] = row["ia"]

    # The image tables must be the same object seen twice: ORB_INDX prints the
    # isc of every supercell orbital, the .nc prints one isc_off per image.
    by_image = {}
    for row in orb["rows"]:
        image = (row["io"] - 1) // dhsdr.no_u
        by_image.setdefault(image, row["isc"])

    mismatches: list[str] = []
    image_counts: set[int] = set()
    for (atom, axis), kinds in sorted(dhsdr.records.items()):
        isc_off = kinds["D_H"].isc_off
        image_counts.add(isc_off.shape[0])
        for image, isc in enumerate(isc_off.tolist()):
            expected = by_image.get(image)
            if expected is None:
                mismatches.append(f"atom {atom} axis {axis}: image {image} is beyond the ORB_INDX supercell")
            elif list(expected) != isc:
                mismatches.append(f"atom {atom} axis {axis}: image {image} isc_off {isc} vs ORB_INDX {expected}")
    checks.append(
        _check(
            "r_vector_table_matches_orb_indx",
            "fail" if mismatches else "pass",
            f"{len(mismatches)} mismatching images: {mismatches[:5]}"
            if mismatches
            else f"{sorted(image_counts)} images, {SUPERCELL_RULE}",
        )
    )
    if image_counts and orb["supercell_images"] not in image_counts:
        checks.append(
            _check(
                "supercell_image_count",
                "fail",
                f"ORB_INDX has {orb['supercell_images']} images, the .nc groups carry {sorted(image_counts)}",
            )
        )
    else:
        checks.append(_check("supercell_image_count", "pass", f"{orb['supercell_images']} images"))

    # The dS filter of calc_dSdR: an element survives only if the row or the
    # column orbital sits on the displaced atom. dH has no such filter.
    leaks: list[str] = []
    for (atom, axis), kinds in sorted(dhsdr.records.items()):
        record = kinds["D_S"]
        touches = (orbital_atom[record.row] == atom) | (orbital_atom[record.col] == atom)
        if not np.all(touches):
            leaks.append(f"atom {atom} axis {axis}: {int((~touches).sum())} of {record.nnz} elements")
    checks.append(
        _check(
            "ds_restricted_to_displaced_atom",
            "fail" if leaks else "pass",
            f"elements on neither the displaced atom's rows nor its columns: {leaks[:5]}"
            if leaks
            else "every dS element has row or column on the displaced atom, as calc_dSdR filters",
        )
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C06: read a SIESTA dHSdR.nc")
    parser.add_argument("--path", type=Path, required=True, help="*.dHSdR.nc to read")
    parser.add_argument("--orb-indx", type=Path, default=None, help="ORB_INDX of the same run")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--no-strict", action="store_true", help="record unknown schema content instead of refusing it")
    args = parser.parse_args(argv)

    dhsdr = read_dhsdr(args.path, orb_indx=args.orb_indx, strict=not args.no_strict)
    report = {
        **dhsdr.metadata(),
        "derivatives": [
            record.summary() for kinds in dhsdr.records.values() for record in kinds.values()
        ],
        "compute_policy": {
            "requested_backend": "cpu",
            "effective_backend": "cpu",
            "reason": "C06 decodes a NetCDF file; there is no numerical workload to place on a GPU",
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if dhsdr.status != INVALID else 1


if __name__ == "__main__":
    raise SystemExit(main())
