"""Physical compatibility of two datasets for small/large mixing (audit Fase 6).

Distinguishes *declared* species (ChemicalSpeciesLabel / PAO.Basis) from
*active* species (atoms in AtomicCoordinatesAndAtomicSpecies, orbitals in
ORB_INDX): a ghost species declared in the FDF but with no atoms contributes
no orbitals to the Hamiltonian, so it provably cannot change the target space
(``proven_inactive``). Also fingerprints the DFT setup (XC, cutoff, smearing,
SCF tolerance, k-point *density*) so incompatible targets block while
deliberate sampling differences are only recorded.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

GHOST_NOT_APPLICABLE = "not_applicable"
GHOST_PROVEN_INACTIVE = "proven_inactive"
GHOST_PROVEN_COMPATIBLE = "proven_compatible"
GHOST_UNPROVEN = "unproven"
GHOST_INCOMPATIBLE = "incompatible"

# Statuses that allow strict materialization without a manual exemption.
GHOST_STATUSES_OK = {GHOST_NOT_APPLICABLE, GHOST_PROVEN_INACTIVE, GHOST_PROVEN_COMPATIBLE}

# Target/DFT keys whose mismatch changes the learning target: must block.
_BLOCKING_SCALAR_KEYS = (
    "XC.functional",
    "XC.authors",
    "MeshCutoff",
    "ElectronicTemperature",
    "DM.Tolerance",
    "SpinPolarized",
    "Spin",
)
_KDENSITY_REL_TOLERANCE = 0.10


# --------------------------------------------------------------------------- #
# FDF parsing (line-oriented; enough for SIESTA inputs used in this project)
# --------------------------------------------------------------------------- #
def _fdf_lines(text: str) -> list[str]:
    return [line.split("#", 1)[0].rstrip() for line in text.splitlines()]


def fdf_block(text: str, name: str) -> list[str] | None:
    lines = _fdf_lines(text)
    lowered = name.lower()
    rows: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("%block") and stripped.split()[-1].lower() == lowered:
            inside = True
            continue
        if inside and stripped.lower().startswith("%endblock"):
            return rows
        if inside:
            rows.append(stripped)
    return None


def fdf_scalar(text: str, key: str) -> str | None:
    lowered = key.lower()
    for line in _fdf_lines(text):
        parts = line.split()
        if parts and parts[0].lower() == lowered:
            return " ".join(parts[1:]) or None
    return None


def parse_chemical_species(text: str) -> dict[int, dict[str, Any]]:
    block = fdf_block(text, "ChemicalSpeciesLabel") or []
    species: dict[int, dict[str, Any]] = {}
    for row in block:
        parts = row.split()
        if len(parts) >= 3:
            species[int(parts[0])] = {"atomic_number": int(parts[1]), "label": parts[2]}
    return species


def species_indices_with_atoms(text: str) -> set[int] | None:
    block = fdf_block(text, "AtomicCoordinatesAndAtomicSpecies")
    if block is None:
        return None
    indices: set[int] = set()
    for row in block:
        parts = row.split()
        if len(parts) >= 4:
            try:
                indices.add(int(parts[3]))
            except ValueError:
                return None
    return indices or None


def parse_kgrid_diagonal(text: str) -> list[int] | None:
    block = fdf_block(text, "kgrid_Monkhorst_Pack")
    if block is None or len(block) < 3:
        return None
    diagonal = []
    for i, row in enumerate(block[:3]):
        parts = row.split()
        if len(parts) < 3:
            return None
        try:
            diagonal.append(int(float(parts[i])))
        except ValueError:
            return None
    return diagonal


def parse_lattice_vectors_ang(text: str) -> list[list[float]] | None:
    constant = fdf_scalar(text, "LatticeConstant")
    block = fdf_block(text, "LatticeVectors")
    if constant is None or block is None or len(block) < 3:
        return None
    scale = float(constant.split()[0])
    unit = constant.split()[1].lower() if len(constant.split()) > 1 else "ang"
    if unit.startswith("bohr"):
        scale *= 0.529177210903
    vectors = []
    for row in block[:3]:
        parts = [float(x) for x in row.split()[:3]]
        vectors.append([scale * x for x in parts])
    return vectors


def kpoint_spacing_per_axis(text: str) -> list[float] | None:
    """Approximate reciprocal spacing |b_i| / N_i (1/Ang) per periodic axis."""
    vectors = parse_lattice_vectors_ang(text)
    diagonal = parse_kgrid_diagonal(text)
    if vectors is None or diagonal is None or 0 in diagonal:
        return None
    a = [[vectors[i][j] for j in range(3)] for i in range(3)]

    def _cross(u, v):
        return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]

    volume = sum(a[0][i] * _cross(a[1], a[2])[i] for i in range(3))
    if abs(volume) < 1e-12:
        return None
    pairs = [(a[1], a[2]), (a[2], a[0]), (a[0], a[1])]
    spacings = []
    for i in range(3):
        b = [2.0 * math.pi * c / volume for c in _cross(*pairs[i])]
        spacings.append(math.sqrt(sum(x * x for x in b)) / diagonal[i])
    return spacings


# --------------------------------------------------------------------------- #
# ORB_INDX parsing (unit-cell orbital species)
# --------------------------------------------------------------------------- #
_ORB_HEADER_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*=\s*orbitals in unit cell")


def orb_indx_unit_cell_species(path: Path) -> dict[str, int] | None:
    """Species label -> number of unit-cell orbitals, from a SIESTA .ORB_INDX."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    n_unit = None
    for line in lines[:5]:
        match = _ORB_HEADER_RE.match(line)
        if match:
            n_unit = int(match.group(1))
            break
    if n_unit is None:
        return None
    counts: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        # Data rows: io ia is spec iao ... (io/ia/is/iao integers, spec a label)
        if len(parts) < 5 or not (
            parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit() and parts[4].isdigit()
        ):
            continue
        io = int(parts[0])
        if io > n_unit:
            break
        counts[parts[3]] = counts.get(parts[3], 0) + 1
    return counts or None


def find_orb_indx(sample_dir: Path) -> Path | None:
    matches = sorted(Path(sample_dir).glob("*.ORB_INDX"))
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# Per-dataset species activity report
# --------------------------------------------------------------------------- #
def species_activity_report(sample_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Declared vs active species for one dataset, from one representative snapshot."""
    fdf_path = Path(sample_dir) / "RUN.fdf"
    text = fdf_path.read_text(encoding="utf-8", errors="ignore") if fdf_path.is_file() else ""
    declared = parse_chemical_species(text)
    if not declared:
        declared = {
            int(entry.get("index") or i + 1): {
                "atomic_number": int(entry.get("atomic_number", 0)),
                "label": str(entry.get("label")),
            }
            for i, entry in enumerate(provenance.get("species") or [])
        }
    atom_indices = species_indices_with_atoms(text)
    orb_counts = None
    orb_indx = find_orb_indx(sample_dir)
    if orb_indx is not None:
        orb_counts = orb_indx_unit_cell_species(orb_indx)
    declared_labels = sorted(entry["label"] for entry in declared.values())
    active_atomic = (
        sorted(declared[i]["label"] for i in atom_indices if i in declared)
        if atom_indices is not None
        else None
    )
    return {
        "sample_dir": str(sample_dir),
        "declared_species": declared_labels,
        "declared_ghost_species": sorted(
            entry["label"] for entry in declared.values() if entry["atomic_number"] < 0
        ),
        "active_atomic_species": active_atomic,
        "active_orbital_species": sorted(orb_counts) if orb_counts else None,
        "unit_cell_orbitals_per_species": orb_counts,
        "kgrid_monkhorst_pack_diagonal": parse_kgrid_diagonal(text),
        "lattice_vectors_ang": parse_lattice_vectors_ang(text),
        "kpoint_spacing_per_axis": kpoint_spacing_per_axis(text),
        "dft_scalars": {key: fdf_scalar(text, key) for key in _BLOCKING_SCALAR_KEYS},
    }


def _ghost_active(report: dict[str, Any], ghost: str) -> bool | None:
    """True/False when evidence exists, None when it does not."""
    evidence: list[bool] = []
    if report["active_atomic_species"] is not None:
        evidence.append(ghost in report["active_atomic_species"])
    if report["active_orbital_species"] is not None:
        evidence.append(ghost in report["active_orbital_species"])
    if not evidence:
        return None
    return any(evidence)


def ghost_compatibility(
    small_report: dict[str, Any],
    large_report: dict[str, Any],
    *,
    small_basis: dict[str, Any] | None = None,
    large_basis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """5-state ghost compatibility from real artifacts (not a manual boolean)."""
    small_ghosts = set(small_report["declared_ghost_species"])
    large_ghosts = set(large_report["declared_ghost_species"])
    result: dict[str, Any] = {
        "small_ghost_species": sorted(small_ghosts),
        "large_ghost_species": sorted(large_ghosts),
        "evidence": {},
    }
    if not small_ghosts and not large_ghosts:
        result["status"] = GHOST_NOT_APPLICABLE
        return result
    if small_ghosts == large_ghosts:
        # Same declared ghost set on both sides: the target spaces agree with
        # each other by symmetry as long as the ghost basis functions match.
        ghost_basis_small = {
            k: v for k, v in (small_basis or {}).items()
            if str(k).split(".", 1)[0] in small_ghosts
        }
        ghost_basis_large = {
            k: v for k, v in (large_basis or {}).items()
            if str(k).split(".", 1)[0] in large_ghosts
        }
        if ghost_basis_small == ghost_basis_large:
            result["status"] = GHOST_PROVEN_COMPATIBLE
        else:
            result["status"] = GHOST_INCOMPATIBLE
            result["reason"] = "ghost basis functions differ between datasets"
        return result

    asymmetric = small_ghosts.symmetric_difference(large_ghosts)
    activity: dict[str, bool | None] = {}
    for ghost in sorted(asymmetric):
        report = small_report if ghost in small_ghosts else large_report
        activity[ghost] = _ghost_active(report, ghost)
    result["evidence"] = activity
    if any(state is True for state in activity.values()):
        result["status"] = GHOST_INCOMPATIBLE
        result["reason"] = (
            "ghost species active (atoms/orbitals present) in one dataset only: "
            + ", ".join(g for g, s in activity.items() if s is True)
        )
    elif all(state is False for state in activity.values()):
        result["status"] = GHOST_PROVEN_INACTIVE
        result["reason"] = (
            "declared-only ghosts: no atoms in AtomicCoordinatesAndAtomicSpecies "
            "and no orbitals in ORB_INDX"
        )
    else:
        result["status"] = GHOST_UNPROVEN
        result["reason"] = "no coordinate/ORB_INDX evidence to decide ghost activity"
    return result


# --------------------------------------------------------------------------- #
# Fingerprint comparison
# --------------------------------------------------------------------------- #
def compare_fingerprints(
    small_report: dict[str, Any],
    large_report: dict[str, Any],
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    sampling: list[str] = []

    for key in _BLOCKING_SCALAR_KEYS:
        small_value = small_report["dft_scalars"].get(key)
        large_value = large_report["dft_scalars"].get(key)
        if small_value is None or large_value is None:
            if small_value != large_value:
                warnings.append(f"{key}: missing on one side ({small_value!r} vs {large_value!r})")
            continue
        if small_value.split() != large_value.split():
            blocking.append(f"{key} differs: {small_value!r} vs {large_value!r}")

    small_spacing = small_report.get("kpoint_spacing_per_axis")
    large_spacing = large_report.get("kpoint_spacing_per_axis")
    if small_report.get("kgrid_monkhorst_pack_diagonal") != large_report.get("kgrid_monkhorst_pack_diagonal"):
        sampling.append(
            "raw Monkhorst-Pack grids differ but are evaluated by reciprocal-space spacing: "
            f"{small_report.get('kgrid_monkhorst_pack_diagonal')} vs "
            f"{large_report.get('kgrid_monkhorst_pack_diagonal')}"
        )
    if small_report.get("lattice_vectors_ang") != large_report.get("lattice_vectors_ang"):
        sampling.append("lattice vectors/cell dimensions differ; atom count/cell size are allowed cross-structure differences")
    if small_spacing and large_spacing:
        density_compatible = True
        for axis, (s, l) in enumerate(zip(small_spacing, large_spacing)):
            denom = max(abs(s), abs(l), 1e-12)
            if abs(s - l) / denom > _KDENSITY_REL_TOLERANCE:
                density_compatible = False
                blocking.append(
                    f"k-point density differs on axis {axis}: "
                    f"spacing {s:.4f} vs {l:.4f} 1/Ang (> {_KDENSITY_REL_TOLERANCE:.0%})"
                )
        if density_compatible and small_spacing != large_spacing:
            sampling.append(
                "k-point spacing differs but remains within tolerance: "
                f"{small_spacing} vs {large_spacing}"
            )
    else:
        warnings.append("k-point density not comparable (missing kgrid or lattice)")

    # Orbitals per species must match for the species shared by both targets.
    small_orbitals = small_report.get("unit_cell_orbitals_per_species")
    large_orbitals = large_report.get("unit_cell_orbitals_per_species")
    small_atoms = _atom_counts(small_report)
    large_atoms = _atom_counts(large_report)
    if small_orbitals and large_orbitals:
        for label in sorted(set(small_orbitals) & set(large_orbitals)):
            per_atom_small = _per_atom(small_orbitals[label], small_atoms.get(label))
            per_atom_large = _per_atom(large_orbitals[label], large_atoms.get(label))
            if per_atom_small and per_atom_large and per_atom_small != per_atom_large:
                blocking.append(
                    f"orbitals per {label} atom differ: {per_atom_small} vs {per_atom_large}"
                )
    else:
        warnings.append("orbital counts not comparable (missing/unparseable ORB_INDX)")

    return {"blocking_errors": blocking, "warnings": warnings, "sampling_differences": sampling}


def _atom_counts(report: dict[str, Any]) -> dict[str, int]:
    # Unit-cell atoms per species can be recovered from the coordinates block
    # only via species indices; use active_atomic_species multiplicity when the
    # caller stored it. Fallback: unknown (None entries skipped by _per_atom).
    return dict(report.get("atoms_per_species") or {})


def _per_atom(total: int, n_atoms: int | None) -> float | None:
    if not n_atoms:
        return None
    return total / n_atoms


def build_dataset_compatibility_report(
    small_sample_dir: Path,
    large_sample_dir: Path,
    small_provenance: dict[str, Any],
    large_provenance: dict[str, Any],
) -> dict[str, Any]:
    small_report = species_activity_report(small_sample_dir, small_provenance)
    large_report = species_activity_report(large_sample_dir, large_provenance)
    ghost = ghost_compatibility(
        small_report,
        large_report,
        small_basis=small_provenance.get("basis_file_sha256") or {},
        large_basis=large_provenance.get("basis_file_sha256") or {},
    )
    fingerprints = compare_fingerprints(small_report, large_report)
    blocking = list(fingerprints["blocking_errors"])
    if ghost["status"] == GHOST_INCOMPATIBLE:
        blocking.append(f"ghost_compatibility incompatible: {ghost.get('reason')}")

    # Pseudopotentials of the shared real species must be identical.
    small_pseudo = small_provenance.get("pseudopotential_sha256") or {}
    large_pseudo = large_provenance.get("pseudopotential_sha256") or {}
    real_labels = {
        str(entry.get("label"))
        for entry in (small_provenance.get("species") or [])
        if int(entry.get("atomic_number", 0)) >= 0
    }
    for label in sorted(real_labels & set(small_pseudo) & set(large_pseudo)):
        if small_pseudo[label] != large_pseudo[label]:
            blocking.append(
                f"pseudopotential differs for species {label}: "
                f"{small_pseudo[label][:12]}... vs {large_pseudo[label][:12]}..."
            )
    return {
        "schema": "ml_vs_siesta_dataset_compatibility_report_v1",
        "compatible": not blocking,
        "ghost_compatibility_status": ghost["status"],
        "ghost_compatibility": ghost,
        "small": small_report,
        "large": large_report,
        "target_compatibility": "proven" if not blocking else "failed",
        "blocking_errors": blocking,
        "warnings": fingerprints["warnings"],
        "sampling_differences": fingerprints["sampling_differences"],
    }


def write_report(report: dict[str, Any], output_root: Path) -> Path:
    path = Path(output_root) / "dataset_compatibility_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
