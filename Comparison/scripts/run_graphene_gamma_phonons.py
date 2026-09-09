#!/usr/bin/env python3
"""C18 / E-F_001-S21: the Gamma phonons of graphene, at two auxiliary FC ranges.

An FC run gives the interatomic force constants only out to the range of the
cell it was run in. At ``Gamma`` that is not obviously a limitation --
``D(Gamma) = sum_l Phi_l``, and a 1x1 run already sums every image implicitly,
because displacing an atom displaces all its periodic images with it. Whether
that implicit sum is the converged one is a measurement, not an argument, and it
is the measurement this script makes: the same physics, the same displacement
and the same basis run in an auxiliary supercell of ``2n+1`` cells per axis, so
that the IFC tail and the Gamma frequencies can be watched as the range widens.

What each range produces, independently and reproducibly:

* one SIESTA FC run of the replicated cell, displacing only the ``na`` unit-cell
  atoms (``FC.First 1`` / ``FC.Last na``) -- the rest of the supercell is there
  to receive forces, not to be displaced;
* the raw IFC, with its distance-resolved shells and the tail ratio;
* the raw *and* the ASR-corrected Gamma modes as two separate artifacts, each
  carrying the raw translational residual, so that "the acoustic modes are zero"
  is never a claim the correction smuggled in;
* a sector report that identifies the acoustic branches by their projection on
  the mass-weighted uniform translations and the E2g doublet by degeneracy and
  in-plane character -- no eigenvector is ever imposed.

The comparison across ranges is the point of the campaign: per-branch frequency
differences in cm^-1 and the tail ratio, published as numbers rather than as a
verdict.

Geometry, basis, pseudopotentials, displacement and every FC directive enter the
run signature, so a range whose inputs are unchanged is reused rather than
recomputed, and a changed displacement or k-grid is a different artifact.

Backend: the same C01 SIESTA binary as every other reference here, and
:func:`run_epc_siesta_reference.backend_preflight` asks it for a GPU path before
any SCF is spent. There is none in this build, and the requested/effective
backend and that evidence travel into the manifest.

The analysis layer of ``materials/graphene/RUN.fdf`` (band lines, PDOS, the
Wannier manifolds) is dropped from the FC input: it is postprocessing whose cost
and band-index assumptions do not survive replication, and it changes no force.
Everything that decides a force -- XC, mesh cutoff, basis, SCF tolerances,
electronic temperature -- is copied unchanged and fingerprinted.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import phonon_provider as pp  # noqa: E402
from artifact_signature import (  # noqa: E402
    PHYSICAL_PHONON,
    epc_artifact_node,
    file_sha256,
    input_signature_sha256,
)
from fdf_materialization import (  # noqa: E402
    _replace_or_append_block,
    extract_fdf_structure,
    materialize_sample_fdf,
)
from orbital_contract import build_orbital_contract  # noqa: E402
from reference_provenance import canonical_sha256  # noqa: E402
from run_epc_siesta_reference import (  # noqa: E402
    backend_preflight,
    canonical_ang_base_fdf,
    physics_fingerprint,
    read_system_label,
)
from run_hamiltonian_derivative_siesta_references import run_siesta  # noqa: E402
from run_inventory import (  # noqa: E402
    SIESTA_RUNTIME_DIR,
    collect_siesta_runtime,
    validate_siesta_runtime_ref,
    write_siesta_runtime_record,
)
from siesta_output_status import parse_siesta_output  # noqa: E402
from siesta_run_fdf import render_fc_layer  # noqa: E402

SCHEMA = "graphene_gamma_phonons_v1"
MANIFEST_NAME = "graphene_gamma_phonon_manifest.json"

DEFAULT_MATERIAL_FDF = REPO_ROOT / "materials/graphene/RUN.fdf"
DEFAULT_BASIS_DIR = REPO_ROOT / "materials/graphene/basis"
DEFAULT_PSEUDO_DIR = REPO_ROOT / "materials/graphene/pseudos"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/epc/phonons/graphene"
DEFAULT_SIESTA_COMMAND = "/home/christian/bin/siesta"

# The amplitude of the FC run certified by S11/C09 for this geometry. It is part
# of the signature: another displacement is another artifact, not a refinement.
DEFAULT_DISPLACEMENT_ANG = 0.01
DEFAULT_RANGES = ((1, 1, 1), (3, 3, 1), (5, 5, 1))

# Postprocessing of the primitive cell that neither enters a force nor survives
# replication (the Wannier manifold names absolute band indices of the 2-atom
# cell). Dropped from the FC input and recorded as dropped.
ANALYSIS_BLOCKS = (
    "BandLines",
    "ProjectedDensityOfStates",
    "Wannier.Manifolds",
    "Wannier.Manifold.entangled",
)
ANALYSIS_DIRECTIVES = ("bandlinesscale", "wannier.", "writemullikenpop")

# Off by default: the matrix derivatives are the GO-2 artifact of the primitive
# reference run (S11), and writing them for every image of a supercell costs
# disk this campaign has no use for. --save-dhs turns them back on.
DEFAULT_SAVE_DHS = False
DEFAULT_DHDR_TOLERANCE = "0.0 Ry/Bohr"
DEFAULT_DSDR_TOLERANCE = "0.0 1/Bohr"

_SCF_ITERATIONS = re.compile(r"SCF cycle converged after\s+(\d+)\s+iterations")
_FC_STEP = re.compile(r"Begin FC step\s*=\s*(-?\d+)")


class GammaPhononError(RuntimeError):
    """The campaign refuses to publish modes it cannot stand behind."""


def _repo_relative(path: Path) -> str:
    """Repository-relative when it can be (an --output-root may live anywhere)."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


# --------------------------------------------------------------------------- #
# fdf layers
# --------------------------------------------------------------------------- #


def drop_analysis_layer(text: str) -> tuple[str, list[str]]:
    """Remove the postprocessing blocks/directives; report what was removed."""
    removed: list[str] = []
    lines = text.splitlines()
    kept: list[str] = []
    closing: str | None = None
    wanted = {name.lower() for name in ANALYSIS_BLOCKS}
    for line in lines:
        clean = line.split("#", 1)[0].strip()
        lowered = clean.lower()
        if closing is not None:
            if lowered.startswith(closing):
                closing = None
            continue
        if lowered.startswith("%block"):
            name = lowered.split(None, 1)[1].strip() if len(lowered.split(None, 1)) > 1 else ""
            if name in wanted:
                closing = "%endblock"
                removed.append(f"%block {name}")
                continue
        # `%PDOS.kgrid_Monkhorst_Pack ... %end ...` is not valid fdf block syntax,
        # so its rows would be left behind as stray directives; it is dropped
        # whole, for the same reason it is inert.
        if lowered.startswith("%pdos"):
            closing = "%end"
            removed.append(clean)
            continue
        first = lowered.split(None, 1)[0] if clean else ""
        if first.startswith(ANALYSIS_DIRECTIVES):
            removed.append(clean)
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n", removed


def scaled_kgrid(repeats: Sequence[int], base_grid: Sequence[int]) -> list[int]:
    """The supercell grid that samples the same reciprocal density as the base.

    ``ceil`` rather than ``round``: erring towards a denser mesh keeps the
    supercell forces at least as converged as the primitive ones. The resulting
    grids are recorded per range, because they are not identical samplings and
    a frequency difference between ranges partly measures that.
    """
    return [max(1, -(-int(count) // int(repeat))) for count, repeat in zip(base_grid, repeats)]


def base_kgrid(text: str) -> list[int]:
    """The diagonal of the base ``kgrid_Monkhorst_Pack`` block."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.split("#", 1)[0].strip().lower().startswith("%block kgrid_monkhorst_pack"):
            rows = [lines[index + 1 + axis].split() for axis in range(3)]
            return [int(rows[axis][axis]) for axis in range(3)]
    raise GammaPhononError("The material fdf declares no kgrid_Monkhorst_Pack block.")


def write_kgrid(text: str, grid: Sequence[int]) -> str:
    rows = [
        " " + "  ".join(str(grid[axis]) if axis == index else "0" for axis in range(3)) + "  0.0"
        for index in range(3)
    ]
    return _replace_or_append_block(text, "kgrid_Monkhorst_Pack", rows)


def write_fc_layer(run_fdf: Path, *, displacement_ang: float, unit_atom_count: int, fc: dict[str, Any]) -> None:
    """Turn the materialised single-point fdf into the FC run of the same physics.

    ``materialize_sample_fdf(single_point=True)`` leaves ``MD.TypeOfRun CG``
    behind and fdf honours the *first* occurrence of a key, so the MD layer is
    dropped before the FC layer is appended rather than overridden after it.
    ``FC.Last`` is the unit-cell atom count, never the supercell's: the rest of
    the atoms are there to receive forces.
    """
    kept = [
        line
        for line in run_fdf.read_text(encoding="utf-8").splitlines()
        if not line.split("#", 1)[0].strip().lower().startswith("md.")
    ]
    layer = render_fc_layer(
        {
            "displacement": f"{displacement_ang:.10g} Ang",
            "first_atom": 1,
            "last_atom": unit_atom_count,
            "save_dhs": bool(fc["save_dhs"]),
            "dHdR_tolerance": fc["dhdr_tolerance"],
            "dSdR_tolerance": fc["dsdr_tolerance"],
        },
        unit_atom_count,
    )
    run_fdf.write_text("\n".join(kept).rstrip() + "\n" + layer, encoding="utf-8")


# --------------------------------------------------------------------------- #
# One range: stage, run, diagonalise
# --------------------------------------------------------------------------- #


def range_tag(repeats: Sequence[int]) -> str:
    return "sc" + "x".join(str(int(value)) for value in repeats)


def stage_range(
    run_dir: Path,
    *,
    base_fdf: Path,
    pseudo_dir: Path,
    repeats: Sequence[int],
    displacement_ang: float,
    fc: dict[str, Any],
) -> dict[str, Any]:
    """Write the supercell RUN.fdf and its pseudopotentials into a clean dir."""
    structure = extract_fdf_structure(base_fdf)
    images = pp.supercell_images(repeats)
    positions, lattice = pp.supercell_geometry(structure.positions_ang, structure.lattice_vectors_ang, images)
    unit_atom_count = structure.atom_count
    species = list(structure.atom_species) * images.shape[0]

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    label = range_tag(repeats)
    materialize_sample_fdf(
        base_fdf,
        run_dir / "RUN.fdf",
        positions_ang=[list(position) for position in positions],
        atom_species=species,
        lattice_vectors_ang=[list(vector) for vector in lattice],
        system_label=label,
        system_name=label,
        single_point=True,
    )
    text, removed = drop_analysis_layer((run_dir / "RUN.fdf").read_text(encoding="utf-8"))
    grid = scaled_kgrid(repeats, base_kgrid(base_fdf.read_text(encoding="utf-8")))
    (run_dir / "RUN.fdf").write_text(write_kgrid(text, grid), encoding="utf-8")
    write_fc_layer(
        run_dir / "RUN.fdf",
        displacement_ang=displacement_ang,
        unit_atom_count=unit_atom_count,
        fc=fc,
    )

    pseudos = sorted(path for path in pseudo_dir.iterdir() if path.suffix in {".psf", ".psml", ".vps"})
    if not pseudos:
        raise GammaPhononError(f"No pseudopotentials in {pseudo_dir}.")
    for source in pseudos:
        shutil.copy2(source, run_dir / source.name)

    return {
        "repeats": [int(value) for value in repeats],
        "cell_images": images.tolist(),
        "unit_atom_count": unit_atom_count,
        "supercell_atom_count": int(positions.shape[0]),
        "supercell_lattice_ang": [list(map(float, vector)) for vector in lattice],
        "kgrid_monkhorst_pack": grid,
        "analysis_layer_removed": removed,
        "materialized_fdf_sha256": file_sha256(run_dir / "RUN.fdf"),
        "physics_fingerprint": physics_fingerprint((run_dir / "RUN.fdf").read_text(encoding="utf-8")),
        "pseudopotential_sha256": {source.name: file_sha256(source) for source in pseudos},
        "fc_directives": {
            "displacement_ang": float(displacement_ang),
            "first_atom": 1,
            "last_atom": unit_atom_count,
            **fc,
        },
    }


def range_signature(staged: dict[str, Any], shared: dict[str, Any]) -> str:
    """Signature of one range: geometry, basis, pseudos, displacement, FC settings."""
    return input_signature_sha256(
        {
            "schema": SCHEMA,
            "repeats": staged["repeats"],
            "cell_images": staged["cell_images"],
            "kgrid_monkhorst_pack": staged["kgrid_monkhorst_pack"],
            "materialized_fdf_sha256": staged["materialized_fdf_sha256"],
            "physics_fingerprint": staged["physics_fingerprint"],
            "pseudopotential_sha256": staged["pseudopotential_sha256"],
            "fc_directives": staged["fc_directives"],
            **shared,
        }
    )


def scf_convergence_log(run_out: Path, *, expected_steps: int) -> dict[str, Any]:
    """Per-FC-step SCF evidence: every displacement must have closed its cycle."""
    text = run_out.read_text(encoding="utf-8", errors="ignore") if run_out.is_file() else ""
    iterations = [int(value) for value in _SCF_ITERATIONS.findall(text)]
    steps = [int(value) for value in _FC_STEP.findall(text)]
    return {
        "fc_steps_started": len(steps),
        "fc_steps_expected": expected_steps,
        "scf_converged_steps": len(iterations),
        "scf_iterations_per_step": iterations,
        "scf_iterations_max": max(iterations) if iterations else None,
        "all_steps_converged": len(iterations) == expected_steps == len(steps),
    }


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def execute_range(run_dir: Path, *, siesta_command: str, expected_steps: int, label: str) -> dict[str, Any]:
    """One SIESTA FC invocation, with its cost and its convergence evidence."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.time()
    record = run_siesta(run_dir, command=siesta_command, use_shell=False)
    wall = time.time() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    status = parse_siesta_output(run_dir / "RUN.out", run_dir / "RUN.fdf")
    convergence = scf_convergence_log(run_dir / "RUN.out", expected_steps=expected_steps)
    fc_file = run_dir / f"{label}.FC"
    problems: list[str] = []
    if int(record["returncode"]) != 0:
        problems.append(f"siesta_returncode_{record['returncode']}")
    if not status.get("job_completed"):
        problems.append("job_not_completed")
    if not status.get("scf_converged") or status.get("explicit_nonconvergence"):
        problems.append("scf_not_converged")
    if not convergence["all_steps_converged"]:
        problems.append("fc_step_without_converged_scf")
    if not fc_file.is_file():
        problems.append("missing_fc_file")
    return {
        "returncode": int(record["returncode"]),
        "wall_seconds": round(wall, 3),
        # ponytail: RUSAGE_CHILDREN.ru_maxrss is a high-water mark over all
        # children, so it only rises; honest per range because ranges are
        # sequential.
        "peak_rss_children_kb": int(after),
        "peak_rss_children_rose_kb": int(max(0, after - before)),
        "disk_bytes": _directory_bytes(run_dir),
        "siesta_output_status": status,
        "scf_convergence": convergence,
        "artifact_sha256": {
            name: file_sha256(run_dir / f"{label}.{name.upper()}")
            for name in ("fc", "orb_indx")
            if (run_dir / f"{label}.{name.upper()}").is_file()
        },
        "problems": problems,
        "certified": not problems,
    }


def geometry_node(structure: Any) -> dict[str, Any]:
    """The ``geometry`` node every phonon artifact of this campaign descends from."""
    labels = {item.index: item.label for item in structure.species}
    return epc_artifact_node(
        "geometry",
        {
            "cell": [list(map(float, vector)) for vector in structure.lattice_vectors_ang],
            "species": [labels[index] for index in structure.atom_species],
            "positions_sha256": canonical_sha256(
                [[round(float(value), 12) + 0.0 for value in position] for position in structure.positions_ang]
            ),
            "coordinate_convention": "cartesian_ang",
        },
    )


def phonons_for_range(
    run_dir: Path,
    *,
    label: str,
    staged: dict[str, Any],
    structure: Any,
    base_fdf: Path,
    geometry: dict[str, Any],
    output_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Raw and ASR-corrected Gamma modes of one range, as two separate artifacts."""
    masses = pp.masses_amu_from_fdf(base_fdf)
    raw = pp.force_constants_from_siesta_supercell_run(
        run_dir / f"{label}.FC",
        images=np.array(staged["cell_images"], dtype=np.int64),
        cell_ang=structure.lattice_vectors_ang,
        positions_ang=structure.positions_ang,
        masses_amu=masses,
        provenance=provenance,
    )
    corrected = raw.apply_asr()

    written: dict[str, Any] = {}
    for policy, force_constants in (("raw", raw), ("asr_corrected", corrected)):
        modes = pp.modes_from_force_constants(
            force_constants,
            (0.0, 0.0, 0.0),
            provider=pp.SIESTA_FC_BACKEND,
            force_source=dict(force_constants.provenance or {}),
            fc_range={
                "repeats": staged["repeats"],
                "cell_images": staged["cell_images"],
                "cell_count": force_constants.cell_count,
                "kgrid_monkhorst_pack": staged["kgrid_monkhorst_pack"],
            },
            geometry_signature=geometry["signature_sha256"],
        )
        node = epc_artifact_node(
            PHYSICAL_PHONON, modes.physical_phonon_fields(), {"geometry": geometry}
        )
        payload = {
            **modes.to_dict(),
            "artifact_node": node,
            "ifc_shells": pp.ifc_shells(force_constants),
            "sector_report": pp.mode_sector_report(modes),
        }
        path = output_dir / f"gamma_modes_{policy}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[policy] = {
            "path": _repo_relative(path),
            "sha256": file_sha256(path),
            "signature_sha256": node["signature_sha256"],
            "asr_policy": modes.asr_policy,
            "asr_residual_ev_ang2": modes.asr_residual_ev_ang2,
            "asr_residual_raw_ev_ang2": modes.asr_residual_raw_ev_ang2,
            "frequencies_ev": modes.frequencies_ev.tolist(),
            "frequencies_cm1": (modes.frequencies_ev / pp.CM1_TO_EV).tolist(),
            "ifc_shells": payload["ifc_shells"],
            "sector_report": payload["sector_report"],
        }
    return written


# --------------------------------------------------------------------------- #
# Across ranges
# --------------------------------------------------------------------------- #


def compare_ranges(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What widening the auxiliary supercell did to the tail and to the modes."""
    usable = [row for row in rows if row.get("certified") and row.get("modes")]
    steps: list[dict[str, Any]] = []
    for previous, current in zip(usable, usable[1:]):
        for policy in ("raw", "asr_corrected"):
            before = np.array(previous["modes"][policy]["frequencies_cm1"])
            after = np.array(current["modes"][policy]["frequencies_cm1"])
            if before.shape != after.shape:
                raise GammaPhononError(
                    f"{previous['range']} and {current['range']} produced {before.size} and "
                    f"{after.size} branches; the Gamma modes of one primitive cell cannot differ in count"
                )
            e2g_before = (previous["modes"][policy]["sector_report"]["e2g_candidate"] or {}).get(
                "mean_frequency_cm1"
            )
            e2g_after = (current["modes"][policy]["sector_report"]["e2g_candidate"] or {}).get(
                "mean_frequency_cm1"
            )
            steps.append(
                {
                    "from_range": previous["range"],
                    "to_range": current["range"],
                    "asr_policy": policy,
                    "delta_cm1_per_branch": (after - before).tolist(),
                    "max_abs_delta_cm1": float(np.abs(after - before).max()),
                    "e2g_cm1": [e2g_before, e2g_after],
                    "e2g_delta_cm1": (
                        None if e2g_before is None or e2g_after is None else float(e2g_after - e2g_before)
                    ),
                    "tail_ratio": [
                        previous["modes"][policy]["ifc_shells"]["tail_ratio"],
                        current["modes"][policy]["ifc_shells"]["tail_ratio"],
                    ],
                    "max_distance_ang": [
                        previous["modes"][policy]["ifc_shells"]["max_distance_ang"],
                        current["modes"][policy]["ifc_shells"]["max_distance_ang"],
                    ],
                }
            )
    return {
        "ranges": [row["range"] for row in usable],
        "range_count": len(usable),
        "at_least_two_ranges": len(usable) >= 2,
        "steps": steps,
        "largest_max_abs_delta_cm1": max((step["max_abs_delta_cm1"] for step in steps), default=None),
        "policy": (
            "reported, not gated: the frequency difference between consecutive ranges and the "
            "IFC tail ratio are the convergence evidence; no threshold is imposed here because "
            "the tolerance a consumer needs depends on what it does with omega"
        ),
    }


# --------------------------------------------------------------------------- #
# Campaign
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def run_campaign(
    *,
    material_fdf: Path = DEFAULT_MATERIAL_FDF,
    basis_dir: Path = DEFAULT_BASIS_DIR,
    pseudo_dir: Path = DEFAULT_PSEUDO_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    siesta_command: str = DEFAULT_SIESTA_COMMAND,
    ranges: Sequence[Sequence[int]] = DEFAULT_RANGES,
    displacement_ang: float = DEFAULT_DISPLACEMENT_ANG,
    save_dhs: bool = DEFAULT_SAVE_DHS,
    dhdr_tolerance: str = DEFAULT_DHDR_TOLERANCE,
    dsdr_tolerance: str = DEFAULT_DSDR_TOLERANCE,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    base_fdf = canonical_ang_base_fdf(material_fdf, output_root / "inputs" / "base_ang.fdf")
    structure = extract_fdf_structure(base_fdf)
    geometry = geometry_node(structure)
    contract = build_orbital_contract(material_fdf.parent.name, base_fdf, basis_dir)

    preflight = backend_preflight(siesta_command)
    runtime = collect_siesta_runtime(siesta_command.split()[0])
    runtime_ref = write_siesta_runtime_record(runtime, SIESTA_RUNTIME_DIR)
    runtime_validation = validate_siesta_runtime_ref(runtime_ref, effective_executable=siesta_command.split()[0])

    fc = {"save_dhs": bool(save_dhs), "dhdr_tolerance": dhdr_tolerance, "dsdr_tolerance": dsdr_tolerance}
    shared = {
        "base_fdf_sha256": file_sha256(base_fdf),
        "geometry_signature": geometry["signature_sha256"],
        "basis_contract_hash": contract.get("basis_contract_hash"),
        "orbital_contract_hash": contract.get("orbital_contract_hash"),
        "basis_file_sha256": {
            path.name: file_sha256(path) for path in sorted(basis_dir.glob("*.ion.xml"))
        },
        "siesta_binary_sha256": runtime_ref.get("binary_sha256"),
        "siesta_runtime_record_sha256": runtime_ref.get("record_sha256"),
    }

    rows: list[dict[str, Any]] = []
    for repeats in ranges:
        tag = range_tag(repeats)
        range_dir = output_root / "ranges" / tag
        run_dir = range_dir / "run"
        record_path = range_dir / "range_record.json"
        label = tag
        expected_steps = 6 * structure.atom_count + 1

        previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else None
        if previous and not overwrite and previous.get("certified") and (run_dir / f"{label}.FC").is_file():
            probe_dir = range_dir / ".probe"
            try:
                probe = range_signature(
                    stage_range(
                        probe_dir,
                        base_fdf=base_fdf,
                        pseudo_dir=pseudo_dir,
                        repeats=repeats,
                        displacement_ang=displacement_ang,
                        fc=fc,
                    ),
                    shared,
                )
            finally:
                shutil.rmtree(probe_dir, ignore_errors=True)
            if probe == previous.get("input_signature_sha256"):
                rows.append({**previous, "status": "reused"})
                print(f"[GAMMA-PHONONS]   reused  {tag}")
                continue

        staged = stage_range(
            run_dir,
            base_fdf=base_fdf,
            pseudo_dir=pseudo_dir,
            repeats=repeats,
            displacement_ang=displacement_ang,
            fc=fc,
        )
        signature = range_signature(staged, shared)
        if dry_run:
            rows.append(
                {
                    "range": tag,
                    "status": "planned",
                    "certified": False,
                    "staged": staged,
                    "input_signature_sha256": signature,
                    "run_dir": str(run_dir),
                }
            )
            print(f"[GAMMA-PHONONS]  planned  {tag}  {staged['supercell_atom_count']} atoms")
            continue

        started_at = datetime.now(timezone.utc).isoformat()
        execution = execute_range(
            run_dir, siesta_command=siesta_command, expected_steps=expected_steps, label=label
        )
        row: dict[str, Any] = {
            "range": tag,
            "repeats": staged["repeats"],
            "status": "ok" if execution["certified"] else "failed",
            "run_dir": str(run_dir),
            "system_label": read_system_label(run_dir / "RUN.fdf"),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "input_signature_sha256": signature,
            "staged": staged,
            **execution,
        }
        if execution["certified"]:
            row["modes"] = phonons_for_range(
                run_dir,
                label=label,
                staged=staged,
                structure=structure,
                base_fdf=base_fdf,
                geometry=geometry,
                output_dir=range_dir,
                provenance={
                    "base_fdf": _repo_relative(base_fdf),
                    "base_fdf_sha256": shared["base_fdf_sha256"],
                    "run_fdf_sha256": staged["materialized_fdf_sha256"],
                    "geometry_signature": geometry["signature_sha256"],
                    "basis_contract_hash": shared["basis_contract_hash"],
                    "orbital_contract_hash": shared["orbital_contract_hash"],
                    "siesta_runtime_record_sha256": shared["siesta_runtime_record_sha256"],
                    "input_signature_sha256": signature,
                },
            )
        _write_json(record_path, row)
        rows.append(row)
        print(
            f"[GAMMA-PHONONS] {row['status']:>8}  {tag}  {staged['supercell_atom_count']} atoms  "
            f"{execution['wall_seconds']}s  {'' if execution['certified'] else execution['problems']}"
        )

    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "material_fdf": str(material_fdf),
        "material_fdf_sha256": file_sha256(material_fdf),
        "base_ang_fdf_sha256": shared["base_fdf_sha256"],
        "basis_dir": str(basis_dir),
        "pseudo_dir": str(pseudo_dir),
        "output_root": str(output_root),
        "siesta_command": siesta_command,
        "siesta_runtime_ref": runtime_ref,
        "siesta_runtime_validation": runtime_validation,
        "compute_policy": {
            **{key: preflight[key] for key in ("requested_backend", "effective_backend", "fallback", "reason")},
            "preflight": preflight,
        },
        "geometry_node": geometry,
        "orbital_contract": {
            "orbital_contract_hash": contract.get("orbital_contract_hash"),
            "basis_contract_hash": contract.get("basis_contract_hash"),
            "atom_count": contract.get("atom_count"),
            "orbital_count": contract.get("orbital_count"),
        },
        "phonon_contract_id": pp.CONTRACT_ID,
        "conventions": pp.CANONICAL_CONVENTIONS.to_dict(),
        "units": dict(pp.UNITS),
        "q_fractional": [0.0, 0.0, 0.0],
        "fc": {
            "displacement_ang": float(displacement_ang),
            "first_atom": 1,
            "last_atom": structure.atom_count,
            "save_dhs": bool(save_dhs),
            "tolerances": {"dHdR": dhdr_tolerance, "dSdR": dsdr_tolerance},
        },
        "signature_inputs": shared,
        "ranges_requested": [list(map(int, repeats)) for repeats in ranges],
        "ranges_certified": len([row for row in rows if row.get("certified")]),
        "ranges_reused": len([row for row in rows if row.get("status") == "reused"]),
        "ranges_failed": len([row for row in rows if row.get("status") == "failed"]),
        "resources": {
            "wall_seconds_total": round(sum(float(row.get("wall_seconds") or 0.0) for row in rows), 3),
            "peak_rss_children_kb": max([int(row.get("peak_rss_children_kb") or 0) for row in rows] or [0]),
            "disk_bytes_total": sum(int(row.get("disk_bytes") or 0) for row in rows),
        },
        "range_comparison": compare_ranges(rows),
        "rows": rows,
    }
    _write_json(output_root / MANIFEST_NAME, manifest)
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material-fdf", type=Path, default=DEFAULT_MATERIAL_FDF)
    parser.add_argument("--basis-dir", type=Path, default=DEFAULT_BASIS_DIR)
    parser.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--siesta-command", default=DEFAULT_SIESTA_COMMAND)
    parser.add_argument(
        "--ranges",
        default=",".join("x".join(map(str, repeats)) for repeats in DEFAULT_RANGES),
        help="Auxiliary FC supercells, e.g. '1x1x1,3x3x1,5x5x1'. Odd repeats only.",
    )
    parser.add_argument("--displacement-ang", type=float, default=DEFAULT_DISPLACEMENT_ANG)
    parser.add_argument("--save-dhs", action="store_true", help="Also write *dHSdR.nc per range.")
    parser.add_argument("--fc-dhdr-tolerance", default=DEFAULT_DHDR_TOLERANCE)
    parser.add_argument("--fc-dsdr-tolerance", default=DEFAULT_DSDR_TOLERANCE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Stage and preflight only; no SCF.")
    return parser


def parse_ranges(text: str) -> list[list[int]]:
    ranges = [[int(value) for value in item.split("x")] for item in text.split(",") if item.strip()]
    if len(ranges) < 2:
        raise GammaPhononError(
            f"{text!r} declares {len(ranges)} auxiliary range(s); the IFC tail and the frequencies "
            "can only be measured against a wider range, so at least two are required"
        )
    return ranges


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = run_campaign(
            material_fdf=args.material_fdf,
            basis_dir=args.basis_dir,
            pseudo_dir=args.pseudo_dir,
            output_root=args.output_root,
            siesta_command=args.siesta_command,
            ranges=parse_ranges(args.ranges),
            displacement_ang=args.displacement_ang,
            save_dhs=args.save_dhs,
            dhdr_tolerance=args.fc_dhdr_tolerance,
            dsdr_tolerance=args.fc_dsdr_tolerance,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (GammaPhononError, pp.PhononContractError) as exc:
        print(f"[GAMMA-PHONONS][ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ranges_certified": manifest["ranges_certified"],
                "ranges_failed": manifest["ranges_failed"],
                "ranges_reused": manifest["ranges_reused"],
                "largest_max_abs_delta_cm1": manifest["range_comparison"]["largest_max_abs_delta_cm1"],
                "wall_seconds_total": manifest["resources"]["wall_seconds_total"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["ranges_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
