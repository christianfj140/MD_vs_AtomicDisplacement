"""Basis/orbital compatibility gate (C03 / E-F_001-S4).

Before any Graph2Mat<->SIESTA number is compared, both sides must agree on
*which* orbital each matrix row is. This module derives that agreement from the
files that define it and refuses to infer anything:

  * ``.ion.xml``  -> the ordered PAO shells of one species, and their cutoffs.
  * ``RUN.fdf``   -> which species exist, which of them carry atoms, the atom
                     order, and the spin treatment.
  * ``ORB_INDX``  -> what SIESTA actually emitted: per-orbital ``n,l,m,z,p``,
                     the ``rc``, and the supercell/image rule for periodic
                     blocks.
  * checkpoint    -> the ``PointBasis`` Graph2Mat was trained with.

Two hashes come out, because two different questions are being asked:

``basis_contract_hash``
    Geometry-independent: the ordered orbital table of every species that
    carries atoms, plus spin and block conventions. graphene, AB and MATBG must
    share this one, and so must the checkpoint's basis, or no comparison
    between them means anything.

``orbital_contract_hash``
    That, plus the per-atom species sequence and the resulting orbital count.
    This is the per-system value an H / derivative / eigenspace artifact caches
    against: it changes when a species, a basis file, the p ordering or the
    ORB_INDX orbital table changes.

Nothing here runs SIESTA, Graph2Mat or a prediction: MATBG's 11164 atoms and
44656 orbitals are checked by counting the FDF and the basis, not by building
an H. torch/graph2mat are imported lazily and only when a checkpoint is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from artifact_signature import input_signature_sha256
from benchmark_manifest import file_sha256
from material_bundle import (
    extract_chemical_species,
    extract_coordinate_species_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ORBITAL_CONTRACT_SCHEMA = "orbital_contract_v1"

VERIFIED = "VERIFIED"
DECLARED_ONLY = "DECLARED_ONLY"
INVALID = "INVALID"

# ORB_INDX prints rc rounded to 3 decimals of a Bohr; the .ion.xml carries the
# full value. Anything above this is a different cutoff, not a print artifact.
RC_BOHR_TOL = 2e-3

# Graph2Mat rebuilds its basis table from the .ion.xml at predict time, so its
# PointBasis.R is whatever the *current* sisl derives from the radial function,
# not the file's rc. A checkpoint trained under another sisl can therefore carry
# a different R for a byte-identical basis. That changes the edge cutoff (the
# neighbour topology), never the orbital identity, so it is reported apart.
R_ANG_TOL = 1e-3

# The orbital order SIESTA emits, certified against ORB_INDX below.
M_ORDER = "ascending_m_from_-l_to_+l_per_shell"
SHELL_ORDER = "ion_xml_paos_file_order"
BLOCK_ORDER = "atoms_in_fdf_coordinate_order__orbitals_in_shell_order"
SUPERCELL_RULE = "center(io) = center(iuo) + sum_i cell_vec(i) * isc(i)"
GRAPH2MAT_BASIS_CONVENTION = "siesta_spherical"

# Declared by run_tbg_pure_graph2mat_campaign.py:prepare_target(); C03 checks
# the geometry and the basis reproduce them, without generating an H.
DEFAULT_SYSTEMS: tuple[dict[str, Any], ...] = (
    {
        "label": "graphene",
        "fdf": "materials/graphene/RUN.fdf",
        "basis_dir": "materials/graphene/basis",
        "expected_atoms": 2,
        "expected_orbitals": 8,
        "orb_indx_glob": "Comparison/datasets/graphene_w90_snapshot_scaling/**/graphene.ORB_INDX",
    },
    {
        "label": "bilayer_graphene_AB",
        "fdf": "materials/bilayer_graphene_AB/RUN.fdf",
        "basis_dir": "materials/graphene_common/basis",
        "expected_atoms": 4,
        "expected_orbitals": 16,
        "orb_indx_glob": "Comparison/datasets/bilayer_graphene_AB_md30/**/bilayer_graphene_AB.ORB_INDX",
    },
    {
        "label": "twisted_bilayer_graphene_1p084549deg",
        "fdf": "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf",
        "basis_dir": "materials/graphene_common/basis",
        "expected_atoms": 11164,
        "expected_orbitals": 44656,
        "orb_indx_glob": None,
    },
)

DEFAULT_CHECKPOINT = (
    "Comparison/results/tbg_registry_spectral_loss/training/checkpoints/"
    "spectral-best-epoch=247-step=03472.ckpt"
)

CONTRACT_DIR = Path("Comparison/results/provenance/epc")


def _check(name: str, outcome: str, detail: str = "") -> dict[str, str]:
    """One recorded check. ``outcome`` is pass | fail | absent."""
    return {"check": name, "outcome": outcome, "detail": detail}


def _status(checks: list[dict[str, str]]) -> str:
    if any(item["outcome"] == "fail" for item in checks):
        return INVALID
    if any(item["outcome"] == "absent" for item in checks):
        return DECLARED_ONLY
    return VERIFIED


def _repo_relative(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# .ion.xml -> ordered orbitals
# ---------------------------------------------------------------------------


def read_ion_xml(path: str | Path) -> dict[str, Any]:
    """The PAO shells of one species, expanded to the orbitals SIESTA emits.

    A ``<orbital>`` in the file is one (n, l, zeta) shell; SIESTA emits its
    ``2l+1`` real orbitals with m running from -l to +l, in file order. That
    expansion is exactly what is certified against ORB_INDX.
    """
    path = Path(path)
    root = ElementTree.parse(path).getroot()

    def _text(tag: str) -> str:
        value = root.findtext(tag)
        return "" if value is None else value.strip()

    atomic_number = int(_text("z"))
    paos = root.find("paos")
    shells: list[dict[str, Any]] = []
    orbitals: list[dict[str, Any]] = []
    for node in [] if paos is None else paos.findall("orbital"):
        l = int(node.attrib["l"])
        shell = {
            "n": int(node.attrib["n"]),
            "l": l,
            "zeta": int(node.attrib["z"]),
            "polarized": bool(int(node.attrib.get("ispol", "0"))),
            "cutoff_bohr": float(node.findtext("radfunc/cutoff") or "nan"),
        }
        shells.append(shell)
        for m in range(-l, l + 1):
            orbitals.append({**shell, "m": m, "iao": len(orbitals) + 1})

    return {
        "label": _text("label"),
        "symbol": _text("symbol"),
        "atomic_number": atomic_number,
        # SIESTA marks a ghost (basis without pseudopotential charge) with a
        # negative Z in the .ion.xml; sisl/Graph2Mat expose it as |Z|, so a
        # ghost-H point type is indistinguishable from a real H one downstream.
        "is_ghost": atomic_number < 0,
        "graph2mat_point_type": abs(atomic_number),
        "valence": float(_text("valence") or "nan"),
        "shells": shells,
        "orbitals": orbitals,
        "orbitals_per_atom": len(orbitals),
        "basis_file": _repo_relative(path),
        "basis_sha256": file_sha256(path),
    }


def graph2mat_basis_tuple(shells: list[dict[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    """The ``PointBasis.basis`` Graph2Mat derives: (n_shells, l, parity) per l."""
    counts: dict[int, int] = {}
    for shell in shells:
        counts[shell["l"]] = counts.get(shell["l"], 0) + 1
    return tuple((counts[l], l, (-1) ** l) for l in sorted(counts))


# ---------------------------------------------------------------------------
# RUN.fdf -> species usage, atom order, spin
# ---------------------------------------------------------------------------


def _fdf_key(token: str) -> str:
    """FDF labels ignore case and any ``. - _``: Spin.Polarized == SpinPolarized."""
    return token.lower().translate(str.maketrans("", "", ".-_"))


fdf_key = _fdf_key  # public: every FDF reader must normalise labels the same way


def _fdf_value(fdf_path: Path, key: str) -> str | None:
    wanted = _fdf_key(key)
    for raw in fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) > 1 and _fdf_key(parts[0]) == wanted:
            return parts[1]
    return None


def fdf_spin(fdf_path: Path) -> dict[str, Any]:
    """Spin treatment and the resulting number of spin blocks per H element."""
    spin = _fdf_value(fdf_path, "Spin")
    polarized = _fdf_value(fdf_path, "SpinPolarized")
    orbit = _fdf_value(fdf_path, "SpinOrbit")
    if spin:
        treatment = spin.strip().lower()
    elif orbit and orbit.strip().upper() in {"T", ".TRUE.", "TRUE", "YES"}:
        treatment = "spin-orbit"
    elif polarized and polarized.strip().upper() in {"T", ".TRUE.", "TRUE", "YES"}:
        treatment = "polarized"
    else:
        treatment = "non-polarized"
    components = {
        "non-polarized": 1,
        "none": 1,
        "polarized": 2,
        "collinear": 2,
        "non-collinear": 4,
        "spin-orbit": 8,
    }.get(treatment)
    return {
        "treatment": treatment,
        "spin_components": components,
        "declared_by": "Spin" if spin else ("SpinOrbit" if orbit else ("SpinPolarized" if polarized else "default")),
    }


# ---------------------------------------------------------------------------
# ORB_INDX -> what SIESTA actually emitted
# ---------------------------------------------------------------------------


def parse_orb_indx(path: str | Path) -> dict[str, Any]:
    """Rows of an ORB_INDX plus the supercell bookkeeping in its header."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header = lines[0].split()
    no_u, no_s = int(header[0]), int(header[1])

    rows: list[dict[str, Any]] = []
    for raw in lines[1:]:
        parts = raw.split()
        # io ia is spec iao n l m z p sym rc isc(3) iuo -> 16 columns
        if len(parts) != 16 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "io": int(parts[0]),
                "ia": int(parts[1]),
                "species_index": int(parts[2]),
                "species_label": parts[3],
                "iao": int(parts[4]),
                "n": int(parts[5]),
                "l": int(parts[6]),
                "m": int(parts[7]),
                "zeta": int(parts[8]),
                "polarized": parts[9] == "T",
                "sym": parts[10],
                "rc_bohr": float(parts[11]),
                "isc": [int(parts[12]), int(parts[13]), int(parts[14])],
                "iuo": int(parts[15]),
            }
        )

    unit_cell = [row for row in rows if row["io"] <= no_u]
    return {
        "path": _repo_relative(path),
        "sha256": file_sha256(path),
        "orbitals_unit_cell": no_u,
        "orbitals_supercell": no_s,
        "rows_parsed": len(rows),
        "unit_cell_rows": unit_cell,
        "rows": rows,
        "supercell_images": (no_s // no_u) if no_u else 0,
        "supercell_rule": SUPERCELL_RULE,
        "rc_units": "Bohr",
    }


def _orb_indx_checks(contract: dict[str, Any], orb: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    expected = contract["orbitals"]
    observed = orb["unit_cell_rows"]

    if orb["rows_parsed"] != orb["orbitals_supercell"]:
        checks.append(
            _check(
                "orb_indx_row_count",
                "fail",
                f"header declares {orb['orbitals_supercell']} supercell orbitals, parsed {orb['rows_parsed']}",
            )
        )
    else:
        checks.append(_check("orb_indx_row_count", "pass", f"{orb['rows_parsed']} rows"))

    if len(observed) != len(expected):
        checks.append(
            _check(
                "orbital_count_matches_orb_indx",
                "fail",
                f"contract derives {len(expected)} orbitals, ORB_INDX unit cell has {len(observed)}",
            )
        )
        return checks
    checks.append(_check("orbital_count_matches_orb_indx", "pass", f"{len(expected)} orbitals"))

    mismatches: list[str] = []
    rc_mismatches: list[str] = []
    for want, got in zip(expected, observed):
        key = ("species_label", "iao", "n", "l", "m", "zeta", "polarized")
        differing = [name for name in key if want[name] != got[name]]
        if want["atom_index"] != got["ia"] or want["species_index"] != got["species_index"]:
            differing.append("atom_or_species_index")
        if differing:
            mismatches.append(f"io={got['io']}:{differing}")
        elif abs(want["cutoff_bohr"] - got["rc_bohr"]) > RC_BOHR_TOL:
            rc_mismatches.append(f"io={got['io']} ion.xml={want['cutoff_bohr']:.4f} ORB_INDX={got['rc_bohr']:.3f}")

    checks.append(
        _check(
            "orbital_order_matches_orb_indx",
            "fail" if mismatches else "pass",
            f"{len(mismatches)} mismatching orbitals: {mismatches[:5]}"
            if mismatches
            else f"identical (n,l,m,zeta,polarization) sequence over {len(expected)} orbitals, "
            f"{M_ORDER}",
        )
    )
    checks.append(
        _check(
            "orbital_cutoffs_match_orb_indx",
            "fail" if rc_mismatches else "pass",
            f"{len(rc_mismatches)} rc mismatches beyond {RC_BOHR_TOL} Bohr: {rc_mismatches[:5]}"
            if rc_mismatches
            else f"every rc within {RC_BOHR_TOL} Bohr of the .ion.xml cutoff",
        )
    )

    # Periodic block convention: every supercell orbital must fold back onto the
    # unit-cell orbital of the same atom-within-cell and the same iao, which is
    # what a real-space H block index means.
    no_u = orb["orbitals_unit_cell"]
    n_atoms = len(contract["atoms"])
    folding: list[str] = []
    for row in orb["rows"]:
        if row["io"] <= no_u:
            if row["iuo"] != row["io"] or row["isc"] != [0, 0, 0]:
                folding.append(f"io={row['io']} unit-cell row has isc={row['isc']} iuo={row['iuo']}")
            continue
        base_atom = (row["ia"] - 1) % n_atoms + 1 if n_atoms else 0
        image = next((item for item in observed if item["ia"] == base_atom and item["iao"] == row["iao"]), None)
        if image is None or image["io"] != row["iuo"]:
            folding.append(f"io={row['io']} folds to iuo={row['iuo']}, expected {image['io'] if image else None}")
    checks.append(
        _check(
            "supercell_image_folding",
            "fail" if folding else "pass",
            f"{len(folding)} inconsistent images: {folding[:5]}"
            if folding
            else f"{orb['supercell_images']} images, {SUPERCELL_RULE}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Graph2Mat checkpoint -> the PointBasis it was trained with
# ---------------------------------------------------------------------------


def read_checkpoint_basis(checkpoint: str | Path) -> dict[str, Any]:
    """The basis table stored inside a Graph2Mat/Lightning checkpoint.

    Weights are never loaded into a model and no structure is processed; this
    only reads the pickled ``basis_table`` and the two hyperparameters that
    describe the matrix layout.
    """
    checkpoint = Path(checkpoint)
    sys.path.insert(0, str(REPO_ROOT / "scripts/torch_serialization_compat"))
    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    table = payload.get("basis_table")
    hparams = payload.get("hyper_parameters", {}) or {}
    points = []
    for point in getattr(table, "basis", []) or []:
        points.append(
            {
                "point_type": int(point.type) if str(point.type).lstrip("-").isdigit() else str(point.type),
                "basis": [list(map(int, shell)) for shell in point.basis],
                "basis_size": int(point.basis_size),
                "basis_convention": str(point.basis_convention),
                "R_ang": [float(value) for value in getattr(point, "R", [])]
                if hasattr(getattr(point, "R", None), "__len__")
                else [float(getattr(point, "R", float("nan")))],
            }
        )
    change_of_basis = getattr(table, "change_of_basis", None)
    return {
        "path": _repo_relative(checkpoint),
        "sha256": file_sha256(checkpoint),
        "point_bases": points,
        "change_of_basis": None if change_of_basis is None else [[float(x) for x in row] for row in change_of_basis],
        "n_matrix_components": hparams.get("n_matrix_components"),
        "trained_basis_files": str(hparams.get("basis_files") or ""),
    }


def _checkpoint_checks(contract: dict[str, Any], ckpt: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    by_type = {point["point_type"]: point for point in ckpt["point_bases"]}
    expected = {species["graph2mat_point_type"]: species for species in contract["species"]}

    missing = sorted(set(expected) - set(by_type))
    extra = sorted(set(by_type) - set(expected))
    checks.append(
        _check(
            "checkpoint_point_types",
            "fail" if missing else "pass",
            f"contract needs {sorted(expected)}, checkpoint has {sorted(by_type)}"
            + (f"; unused checkpoint types {extra}" if extra else ""),
        )
    )

    for point_type, species in sorted(expected.items()):
        point = by_type.get(point_type)
        if point is None:
            continue
        want_basis = [list(shell) for shell in graph2mat_basis_tuple(species["shells"])]
        if point["basis"] != want_basis or point["basis_size"] != species["orbitals_per_atom"]:
            checks.append(
                _check(
                    f"checkpoint_basis_{species['label']}",
                    "fail",
                    f"checkpoint basis={point['basis']} size={point['basis_size']}, "
                    f".ion.xml gives {want_basis} size={species['orbitals_per_atom']}",
                )
            )
        else:
            checks.append(
                _check(
                    f"checkpoint_basis_{species['label']}",
                    "pass",
                    f"{want_basis}, {species['orbitals_per_atom']} orbitals/atom",
                )
            )
        if point["basis_convention"] != GRAPH2MAT_BASIS_CONVENTION:
            checks.append(
                _check(
                    f"checkpoint_basis_convention_{species['label']}",
                    "fail",
                    f"{point['basis_convention']!r}; blocks are only comparable to SIESTA in "
                    f"{GRAPH2MAT_BASIS_CONVENTION!r}",
                )
            )
        else:
            checks.append(
                _check(
                    f"checkpoint_basis_convention_{species['label']}",
                    "pass",
                    GRAPH2MAT_BASIS_CONVENTION,
                )
            )

        # Reported, never fatal: R is derived from the radial function by the
        # sisl of the day, so it can drift for a byte-identical basis. It sets
        # the edge cutoff, so a drift changes the neighbour topology (GO-3),
        # not which orbital a row is.
        current = species.get("graph2mat_R_ang")
        if current is None:
            checks.append(
                _check(
                    f"checkpoint_radial_cutoff_{species['label']}",
                    "absent",
                    "graph2mat/sisl unavailable, cannot derive the current PointBasis.R",
                )
            )
        elif max(abs(a - b) for a, b in zip(current, point["R_ang"])) > R_ANG_TOL:
            checks.append(
                _check(
                    f"checkpoint_radial_cutoff_{species['label']}",
                    "absent",
                    f"checkpoint R={point['R_ang']} A, current basis table R={current} A; same orbital "
                    "identity, different Graph2Mat edge cutoff -> re-hash the neighbour topology before "
                    "reusing a derivative",
                )
            )
        else:
            checks.append(
                _check(f"checkpoint_radial_cutoff_{species['label']}", "pass", f"R={current} A")
            )

    spin = contract["spin"]["spin_components"]
    components = ckpt.get("n_matrix_components")
    if components is None:
        checks.append(_check("checkpoint_matrix_components", "absent", "checkpoint declares no n_matrix_components"))
    elif spin is not None and int(components) != int(spin):
        checks.append(
            _check(
                "checkpoint_matrix_components",
                "fail",
                f"checkpoint predicts {components} matrix component(s), FDF spin treatment "
                f"{contract['spin']['treatment']!r} needs {spin}",
            )
        )
    else:
        checks.append(
            _check("checkpoint_matrix_components", "pass", f"{components} component(s), spin {contract['spin']['treatment']!r}")
        )
    return checks


def _current_point_basis_R(basis_dir: Path) -> dict[str, list[float]] | None:
    """PointBasis.R the installed graph2mat/sisl derives now, or None."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts/torch_serialization_compat"))
        from torch_safe_globals import allow_graph2mat_checkpoint_globals

        allow_graph2mat_checkpoint_globals()
        from graph2mat import AtomicTableWithEdges
    except ImportError:
        return None
    try:
        table = AtomicTableWithEdges.from_basis_glob(Path(basis_dir).glob("*.ion.xml"))
    except Exception:  # noqa: BLE001 - a probe must not break the gate
        return None
    return {str(point.type): [float(value) for value in point.R] for point in table.basis}


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------


def build_orbital_contract(
    label: str,
    fdf_path: str | Path,
    basis_dir: str | Path,
    *,
    expected_atoms: int | None = None,
    expected_orbitals: int | None = None,
    probe_graph2mat: bool = False,
) -> dict[str, Any]:
    """Derive the orbital contract of one system from its FDF and .ion.xml."""
    fdf_path = Path(fdf_path)
    basis_dir = Path(basis_dir)

    declared = extract_chemical_species(fdf_path)
    atom_species = extract_coordinate_species_indices(fdf_path)
    populated = {index: atom_species.count(index) for index in sorted(set(atom_species))}
    # Only probed when a checkpoint has to be compared against it; deriving it
    # costs a graph2mat (hence torch) import.
    current_R = _current_point_basis_R(basis_dir) if probe_graph2mat else None

    species: list[dict[str, Any]] = []
    unpopulated: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    by_index: dict[int, dict[str, Any]] = {}
    for entry in declared:
        basis_file = basis_dir / f"{entry.label}.ion.xml"
        if not basis_file.is_file():
            problems.append(
                _check(
                    f"basis_file_{entry.label}",
                    "fail",
                    f"species {entry.label!r} is declared in {_repo_relative(fdf_path)} but "
                    f"{_repo_relative(basis_file)} does not exist",
                )
            )
            continue
        record = read_ion_xml(basis_file)
        record["species_index"] = entry.index
        record["fdf_atomic_number"] = entry.atomic_number
        record["atom_count"] = populated.get(entry.index, 0)
        if current_R is not None:
            record["graph2mat_R_ang"] = current_R.get(str(record["graph2mat_point_type"]))
        if record["label"] != entry.label:
            problems.append(
                _check(
                    f"basis_label_{entry.label}",
                    "fail",
                    f"{_repo_relative(basis_file)} declares label {record['label']!r}",
                )
            )
        if record["atomic_number"] != entry.atomic_number:
            problems.append(
                _check(
                    f"basis_atomic_number_{entry.label}",
                    "fail",
                    f"FDF declares Z={entry.atomic_number} for {entry.label!r}, "
                    f".ion.xml declares Z={record['atomic_number']}",
                )
            )
        # Graph2Mat groups shells by l; if the file does not already list them in
        # ascending l, its block layout and SIESTA's differ silently.
        ls = [shell["l"] for shell in record["shells"]]
        if ls != sorted(ls):
            problems.append(
                _check(
                    f"shells_ascending_l_{entry.label}",
                    "fail",
                    f"{_repo_relative(basis_file)} lists l={ls}; Graph2Mat would reorder them and its "
                    "blocks would no longer match the SIESTA row order",
                )
            )
        if record["atom_count"]:
            species.append(record)
            by_index[entry.index] = record
        else:
            unpopulated.append(record)

    atoms = [
        {
            "atom_index": position + 1,
            "species_index": index,
            "species_label": by_index[index]["label"] if index in by_index else None,
        }
        for position, index in enumerate(atom_species)
    ]

    orbitals: list[dict[str, Any]] = []
    for atom in atoms:
        record = by_index.get(atom["species_index"])
        if record is None:
            continue
        for orbital in record["orbitals"]:
            orbitals.append(
                {
                    "io": len(orbitals) + 1,
                    "atom_index": atom["atom_index"],
                    "species_index": atom["species_index"],
                    "species_label": record["label"],
                    **orbital,
                }
            )

    spin = fdf_spin(fdf_path)
    conventions = {
        "shell_order": SHELL_ORDER,
        "m_order": M_ORDER,
        "block_order": BLOCK_ORDER,
        "supercell_rule": SUPERCELL_RULE,
        "graph2mat_basis_convention": GRAPH2MAT_BASIS_CONVENTION,
        "cutoff_units": "Bohr",
    }

    # Geometry-independent: what "the same orbital" means for these species.
    basis_body = {
        "species": [
            {
                "label": record["label"],
                "atomic_number": record["atomic_number"],
                "is_ghost": record["is_ghost"],
                "orbitals_per_atom": record["orbitals_per_atom"],
                "orbitals": [
                    {key: orbital[key] for key in ("iao", "n", "l", "m", "zeta", "polarized", "cutoff_bohr")}
                    for orbital in record["orbitals"]
                ],
                "basis_sha256": record["basis_sha256"],
            }
            for record in sorted(species, key=lambda item: item["label"])
        ],
        "spin": spin,
        "conventions": conventions,
    }

    contract: dict[str, Any] = {
        "schema": ORBITAL_CONTRACT_SCHEMA,
        "label": label,
        "fdf": _repo_relative(fdf_path),
        "fdf_sha256": file_sha256(fdf_path),
        "basis_dir": _repo_relative(basis_dir),
        "species": species,
        "declared_unpopulated_species": unpopulated,
        "atoms": atoms,
        "atom_count": len(atoms),
        "orbitals": orbitals,
        "orbital_count": len(orbitals),
        "orbitals_per_species": {record["label"]: record["orbitals_per_atom"] for record in species},
        "spin": spin,
        "conventions": conventions,
        "basis_contract_hash": input_signature_sha256(basis_body),
        "problems": problems,
    }
    contract["orbital_contract_hash"] = input_signature_sha256(
        {
            "basis_contract_hash": contract["basis_contract_hash"],
            "atom_species_sequence": [atom["species_label"] for atom in atoms],
            "orbital_count": len(orbitals),
        }
    )
    contract["expected"] = {"atoms": expected_atoms, "orbitals": expected_orbitals}
    return contract


def certify_system(
    label: str,
    fdf_path: str | Path,
    basis_dir: str | Path,
    *,
    expected_atoms: int | None = None,
    expected_orbitals: int | None = None,
    orb_indx: str | Path | None = None,
    checkpoint_basis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the contract and record every check it survives."""
    contract = build_orbital_contract(
        label,
        fdf_path,
        basis_dir,
        expected_atoms=expected_atoms,
        expected_orbitals=expected_orbitals,
        probe_graph2mat=checkpoint_basis is not None,
    )
    checks: list[dict[str, str]] = list(contract.pop("problems"))

    declared_atoms = _fdf_value(Path(fdf_path), "NumberOfAtoms")
    if declared_atoms is not None and int(declared_atoms) != contract["atom_count"]:
        checks.append(
            _check(
                "fdf_atom_count",
                "fail",
                f"NumberOfAtoms={declared_atoms}, coordinate block has {contract['atom_count']}",
            )
        )
    else:
        checks.append(_check("fdf_atom_count", "pass", f"{contract['atom_count']} atoms"))

    if expected_atoms is not None:
        checks.append(
            _check(
                "expected_atom_count",
                "pass" if contract["atom_count"] == expected_atoms else "fail",
                f"expected {expected_atoms}, counted {contract['atom_count']} from the coordinate block",
            )
        )
    if expected_orbitals is not None:
        counted = contract["orbital_count"]
        breakdown = " + ".join(
            f"{sum(1 for atom in contract['atoms'] if atom['species_label'] == record['label'])}x"
            f"{record['orbitals_per_atom']}({record['label']})"
            for record in contract["species"]
        )
        checks.append(
            _check(
                "expected_orbital_count",
                "pass" if counted == expected_orbitals else "fail",
                f"expected {expected_orbitals}, counted {counted} = {breakdown} (no H generated)",
            )
        )

    for record in contract["declared_unpopulated_species"]:
        # graphene declares Ghost-H in ChemicalSpeciesLabel and PAO.Basis but
        # places no ghost atom, so SIESTA emits none of its orbitals. The
        # .ion.xml sitting in the basis directory is still picked up by
        # Graph2Mat's basis glob and becomes a point type SIESTA never emits.
        checks.append(
            _check(
                f"unpopulated_species_{record['label']}",
                "pass",
                f"{record['label']!r} (Z={record['atomic_number']}, ghost={record['is_ghost']}) is declared "
                f"but carries 0 atoms: it contributes 0 of the {contract['orbital_count']} orbitals; its "
                f"{_repo_relative(record['basis_file'])} still adds Graph2Mat point type "
                f"{record['graph2mat_point_type']} to any basis glob over {contract['basis_dir']}",
            )
        )

    if orb_indx is None:
        checks.append(
            _check(
                "orb_indx_certification",
                "absent",
                "no ORB_INDX for this system; the orbital table is derived from .ion.xml + FDF only",
            )
        )
        orb_record = None
    else:
        orb_record = parse_orb_indx(orb_indx)
        checks.extend(_orb_indx_checks(contract, orb_record))
        # Fold the certified table into the per-system hash: a different
        # ORB_INDX orbital table is a different contract.
        contract["orbital_contract_hash"] = input_signature_sha256(
            {
                "orbital_contract_hash": contract["orbital_contract_hash"],
                "orb_indx_table": [
                    [row["io"], row["ia"], row["species_index"], row["species_label"], row["iao"], row["n"],
                     row["l"], row["m"], row["zeta"], row["polarized"], row["sym"], row["rc_bohr"],
                     row["isc"], row["iuo"]]
                    for row in orb_record["rows"]
                ],
            }
        )
        orb_record = {key: value for key, value in orb_record.items() if key not in {"rows", "unit_cell_rows"}}

    if checkpoint_basis is None:
        checks.append(_check("checkpoint_certification", "absent", "no Graph2Mat checkpoint supplied"))
    else:
        checks.extend(_checkpoint_checks(contract, checkpoint_basis))

    return {
        **contract,
        # The full per-orbital table is reproducible from the inputs and would
        # dominate the artifact for MATBG; the hash covers it.
        "orbitals": contract["orbitals"][: 4 * 8],
        "orbitals_truncated": len(contract["orbitals"]) > 4 * 8,
        "atoms": contract["atoms"][:8],
        "atoms_truncated": len(contract["atoms"]) > 8,
        "orb_indx": orb_record,
        "checks": checks,
        "status": _status(checks),
    }


def certify_systems(
    systems: tuple[dict[str, Any], ...] = DEFAULT_SYSTEMS,
    *,
    repo_root: Path = REPO_ROOT,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Certify every system and assert they share one basis contract."""
    checkpoint_basis = None
    checkpoint_error = None
    if checkpoint is not None:
        try:
            checkpoint_basis = read_checkpoint_basis(checkpoint)
        except Exception as exc:  # noqa: BLE001 - a missing torch must not break the gate
            checkpoint_error = f"{type(exc).__name__}: {exc}"

    records = []
    cross_checks: list[dict[str, str]] = []
    for system in systems:
        fdf = repo_root / system["fdf"]
        if not fdf.is_file():
            # A system whose inputs are absent is not "compatible by default".
            cross_checks.append(
                _check(f"system_inputs_{system['label']}", "absent", f"{system['fdf']} is not in this checkout")
            )
            continue
        glob_pattern = system.get("orb_indx_glob")
        orb_indx = None
        if glob_pattern:
            matches = sorted(repo_root.glob(glob_pattern))
            orb_indx = matches[0] if matches else None
        records.append(
            certify_system(
                system["label"],
                repo_root / system["fdf"],
                repo_root / system["basis_dir"],
                expected_atoms=system.get("expected_atoms"),
                expected_orbitals=system.get("expected_orbitals"),
                orb_indx=orb_indx,
                checkpoint_basis=checkpoint_basis,
            )
        )

    shared = {record["basis_contract_hash"] for record in records}
    cross_checks.append(
        _check(
            "single_basis_contract_across_systems",
            "pass" if len(shared) == 1 else "fail",
            "; ".join(f"{record['label']}={record['basis_contract_hash'][:12]}" for record in records),
        )
    )
    if checkpoint_error:
        cross_checks.append(_check("checkpoint_readable", "absent", checkpoint_error))

    statuses = [record["status"] for record in records]
    status = (
        INVALID
        if INVALID in statuses or any(item["outcome"] == "fail" for item in cross_checks)
        else (DECLARED_ONLY if DECLARED_ONLY in statuses or any(item["outcome"] == "absent" for item in cross_checks) else VERIFIED)
    )
    return {
        "schema": ORBITAL_CONTRACT_SCHEMA,
        "status": status,
        "basis_contract_hash": sorted(shared)[0] if len(shared) == 1 else None,
        "systems": records,
        "checkpoint": checkpoint_basis,
        "cross_checks": cross_checks,
        "compute_policy": {
            "requested_backend": "cpu",
            "effective_backend": "cpu",
            "reason": "C03 parses XML/text and hashes; there is no numerical workload to place on a GPU",
        },
    }


def compare_contracts(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """What changed between two contracts, and whether reuse survives it."""
    reasons = []
    for field in ("basis_contract_hash", "orbital_contract_hash"):
        if previous.get(field) != current.get(field):
            reasons.append(field)
    return {
        "compatible": not reasons,
        "changed": reasons,
        "previous": {field: previous.get(field) for field in ("basis_contract_hash", "orbital_contract_hash")},
        "current": {field: current.get(field) for field in ("basis_contract_hash", "orbital_contract_hash")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C03 basis/orbital compatibility gate")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Graph2Mat checkpoint to certify")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / CONTRACT_DIR / "orbital_contract.json")
    args = parser.parse_args(argv)

    checkpoint = None if args.no_checkpoint else REPO_ROOT / args.checkpoint
    if checkpoint is not None and not checkpoint.is_file():
        checkpoint = None
    report = certify_systems(checkpoint=checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": _repo_relative(args.output),
                "status": report["status"],
                "basis_contract_hash": report["basis_contract_hash"],
                "systems": {
                    record["label"]: {
                        "status": record["status"],
                        "atoms": record["atom_count"],
                        "orbitals": record["orbital_count"],
                        "orbital_contract_hash": record["orbital_contract_hash"],
                        "failed": [item["check"] for item in record["checks"] if item["outcome"] == "fail"],
                        "unverified": [item["check"] for item in record["checks"] if item["outcome"] == "absent"],
                    }
                    for record in report["systems"]
                },
                "cross_checks": report["cross_checks"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] != INVALID else 1


if __name__ == "__main__":
    raise SystemExit(main())
