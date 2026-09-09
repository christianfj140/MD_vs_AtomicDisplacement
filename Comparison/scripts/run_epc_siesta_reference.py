#!/usr/bin/env python3
"""C09 / E-F_001-S11: the SIESTA reference of graphene -- ``FC.Save.dHS`` and central ``R +- delta v``.

GO-2 asks one question: do the derivatives SIESTA writes into ``*dHSdR.nc``
reproduce a finite difference of *the same* H and S? This script produces both
sides of that comparison from one frozen set of physics inputs, and nothing
else. It computes no derivative and no metric; C09 does that from these
artifacts.

Three branches, all materialised from the same base fdf:

``equilibrium``
    one single-point SCF at ``R``. The geometry, the orbital mapping and the
    H/S that every other branch is measured against.
``fc_dhs``
    one FC run *per swept delta*, with ``FC.Displacement = delta``. Each writes
    ``*.FC``, ``*dHSdR.nc`` -- SIESTA's own ``dH/dR``, ``dS/dR`` at half-step
    ``delta`` -- and, for free, the ``<label>.<atom>-<disp>.TSHS`` of every
    internal +-displacement. One FC run per delta rather than one FC run at
    all: the FC half-step *is* the amplitude, so "``D_H`` and ``D_S`` comparable
    for each delta" only means something if this branch is swept too.
``fd_central``
    single-point SCF at ``R +- delta v`` for every direction of
    :func:`fd_perturbation_space.default_direction_set` (one Cartesian, one
    collective, one uniform translation) and every swept delta -- the
    *independent* finite difference: our displaced geometry, our fdf, SIESTA's
    SCF.

What "the same H and S" is made to mean, mechanically:

* one base fdf for every branch, so mesh, k-grid, XC, SCF tolerances, basis and
  pseudopotentials are copies rather than re-declarations;
  :func:`physics_fingerprint` hashes every fdf line that is not geometry, label
  or the MD/FC layer, and a branch that disagrees is refused;
* the pseudopotentials are copied from one directory and hashed per run;
* every run writes an ORB_INDX, hashed and compared against the equilibrium
  one: a run whose orbital mapping differs cannot be certified, because its
  matrix rows do not mean the same thing;
* the matrix artifact is the TSHS, never the HSX: HSX is single precision and
  the C01 follow-up list forbids it as the sole elementwise FD reference.

Certification is fail-closed. A run joins ``certification_set`` only when SIESTA
reported a completed job *and* SCF convergence, the required artifacts exist,
the ORB_INDX matches the equilibrium one and -- for the single-point branches --
the geometry stored in the TSHS is still the one that was asked for (an
inherited MD block would have evolved it). A delta is usable only when both
members of its ``+-`` pair are certified.

Backend: this SIESTA has no GPU build. :func:`backend_preflight` asks the binary
before any SCF is spent and records requested/effective backend with evidence.

Restart: every run directory keeps a ``run_record.json`` carrying the signature
of its inputs. A rerun whose signature matches and whose artifacts are intact is
reused, so a compatible ground state is never recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from fdf_materialization import (  # noqa: E402
    _replace_or_append_block,
    _set_fdf_directive,
    extract_fdf_structure,
    materialize_sample_fdf,
)
from material_bundle import read_fdf_block  # noqa: E402
from orbital_contract import build_orbital_contract  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    reference_output_geometry_error,
    run_siesta,
)
from run_inventory import (  # noqa: E402
    SIESTA_RUNTIME_DIR,
    collect_siesta_runtime,
    validate_siesta_runtime_ref,
    write_siesta_runtime_record,
)
from siesta_output_status import parse_siesta_output  # noqa: E402
from siesta_run_fdf import render_fc_layer  # noqa: E402

SCHEMA = "epc_siesta_reference_v1"

BRANCH_EQUILIBRIUM = "equilibrium"
BRANCH_FC = "fc_dhs"
BRANCH_FD = "fd_central"

DEFAULT_MATERIAL_FDF = REPO_ROOT / "materials/graphene/RUN.fdf"
DEFAULT_BASIS_DIR = REPO_ROOT / "materials/graphene/basis"
DEFAULT_PSEUDO_DIR = REPO_ROOT / "materials/graphene/pseudos"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
DEFAULT_SIESTA_COMMAND = "/home/christian/bin/siesta"

# Sparsification off by default: FC and FD must be compared on the same
# elements, and the reader (C06) already records that dHdR.Tolerance is a no-op
# in this release while dSdR.Tolerance is not. S12 re-runs with other values;
# they are part of the run signature, so a changed tolerance is a new run.
DEFAULT_DHDR_TOLERANCE = "0.0 Ry/Bohr"
DEFAULT_DSDR_TOLERANCE = "0.0 1/Bohr"

# Substrings that would mean this build can offload; absence is the evidence
# that CPU is not merely the default but the only scientifically available path.
GPU_BUILD_TOKENS = ("cuda", "gpu", "openacc", "hip ", "rocm", "magma")

# fdf lines that legitimately differ between branches and must not enter the
# physics fingerprint: the geometry, its labels, and the MD/FC layer.
_VARIABLE_KEYS = {"systemname", "systemlabel", "numberofatoms", "atomiccoordinatesformat"}
_VARIABLE_BLOCKS = {"atomiccoordinatesandatomicspecies"}
_VARIABLE_PREFIXES = ("md.", "fc.")

STATUS_FIELDS = [
    "run_id",
    "branch",
    "status",
    "certified",
    "direction",
    "delta_ang",
    "sign_label",
    "scf_converged",
    "job_completed",
    "wall_seconds",
    "returncode",
    "error",
    "run_dir",
]


class SiestaReferenceError(RuntimeError):
    """The reference campaign refuses to produce an uncertifiable artifact."""


# --------------------------------------------------------------------------- #
# fdf helpers
# --------------------------------------------------------------------------- #


def _fdf_lines(text: str) -> list[tuple[str, str]]:
    """``(key, normalised line)`` for every directive line, comments stripped."""
    rows: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key = line.split(None, 1)[0].strip().lower()
        rows.append((key, " ".join(line.split())))
    return rows


def physics_fingerprint(text: str) -> str:
    """Hash of everything in an fdf that is *not* geometry, label or MD/FC layer.

    Two branches with the same fingerprint ran the same physics: same mesh, same
    k-grid, same XC, same SCF tolerances, same basis block, same output flags.

    Byte-identical repetitions of a line collapse -- appending the FC layer
    restates ``TS.HS.Save T``, which fdf resolves to the same value it already
    had. A repetition with a *different* value does not collapse: that is a
    real ambiguity and must change the hash.
    """
    kept: list[str] = []
    seen: set[str] = set()
    in_variable_block = False
    for key, line in _fdf_lines(text):
        if key in {"%block", "%endblock"}:
            name = line.split(None, 1)[1].strip().lower() if len(line.split(None, 1)) > 1 else ""
            if name in _VARIABLE_BLOCKS:
                in_variable_block = key == "%block"
                continue
            in_variable_block = False
        if in_variable_block:
            continue
        if key in _VARIABLE_KEYS or key.startswith(_VARIABLE_PREFIXES):
            continue
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(normalized)
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def canonical_ang_base_fdf(material_fdf: Path, output_fdf: Path) -> Path:
    """Rewrite the material fdf with explicit ``Ang`` coordinates.

    ``materials/graphene/RUN.fdf`` is ``Fractional`` and
    ``shared/fdf_materialization`` accepts ``Ang`` only, deliberately: an
    implicit unit is how a stencil silently displaces the wrong distance. sisl
    reads the geometry in the units the file declares; from here on every branch
    shares one Ang base and the species column is the file's own.
    """
    import sisl

    geometry = sisl.get_sile(str(material_fdf)).read_geometry()
    species = [int(row.split()[3]) for row in read_fdf_block(material_fdf, "AtomicCoordinatesAndAtomicSpecies")]
    if len(species) != len(geometry):
        raise SiestaReferenceError(
            f"{material_fdf}: {len(species)} coordinate rows but sisl read {len(geometry)} atoms."
        )
    text = material_fdf.read_text(encoding="utf-8")
    text = _set_fdf_directive(text, "AtomicCoordinatesFormat", "Ang")
    text = _set_fdf_directive(text, "LatticeConstant", "1.0 Ang")
    text = _replace_or_append_block(
        text,
        "LatticeVectors",
        [" " + "  ".join(f"{value:.12f}" for value in vector) for vector in geometry.lattice.cell],
    )
    text = _replace_or_append_block(
        text,
        "AtomicCoordinatesAndAtomicSpecies",
        [
            " " + "  ".join(f"{value:.12f}" for value in position) + f"  {index}"
            for position, index in zip(geometry.xyz, species)
        ],
    )
    output_fdf.parent.mkdir(parents=True, exist_ok=True)
    output_fdf.write_text(text, encoding="utf-8")
    return output_fdf


def write_fc_layer(run_fdf: Path, *, displacement_ang: float, atom_count: int, tolerances: dict[str, str]) -> None:
    """Turn a materialised single-point fdf into the FC run of the same physics.

    The single-point materialisation leaves ``MD.TypeOfRun CG`` behind and fdf
    honours the *first* occurrence of a key, so the MD layer is dropped before
    the FC layer is appended rather than overridden after it.
    """
    kept = [line for line in run_fdf.read_text(encoding="utf-8").splitlines() if not _is_md_line(line)]
    layer = render_fc_layer(
        {
            "displacement": f"{displacement_ang:.10g} Ang",
            "first_atom": 1,
            "last_atom": atom_count,
            "save_dhs": True,
            "dHdR_tolerance": tolerances["dhdr"],
            "dSdR_tolerance": tolerances["dsdr"],
        },
        atom_count,
    )
    run_fdf.write_text("\n".join(kept).rstrip() + "\n" + layer, encoding="utf-8")


def _is_md_line(line: str) -> bool:
    clean = line.split("#", 1)[0].strip()
    return bool(clean) and clean.split(None, 1)[0].lower().startswith("md.")


def read_system_label(run_fdf: Path) -> str:
    for key, line in _fdf_lines(run_fdf.read_text(encoding="utf-8")):
        if key == "systemlabel":
            return line.split(None, 1)[1].strip()
    raise SiestaReferenceError(f"{run_fdf}: no SystemLabel.")


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def backend_preflight(siesta_command: str) -> dict[str, Any]:
    """Ask the binary whether a GPU path exists, before an SCF is spent on it.

    The compute policy wants GPU whenever a scientifically equivalent backend
    exists. SIESTA's own ``--version`` banner lists its parallelisations and
    optional packages; no GPU token there means the CPU path is not a default
    but the only one, and that evidence is what travels into the manifest.
    """
    argv = shlex.split(siesta_command) + ["--version"]
    started = time.time()
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False, stdin=subprocess.DEVNULL
        )
        # Banner from stdout only: stderr carries the X11/MPI chatter of the
        # environment, not a statement about the build.
        banner = completed.stdout or ""
        returncode: int | None = completed.returncode
        stderr, error = (completed.stderr or "").strip(), ""
    except Exception as exc:  # noqa: BLE001 - a failed probe is recorded, not raised
        banner, returncode, stderr, error = "", None, "", repr(exc)
    found = sorted({token.strip() for token in GPU_BUILD_TOKENS if token in banner.lower()})
    return {
        "probe_argv": argv,
        "probe_returncode": returncode,
        "probe_seconds": round(time.time() - started, 3),
        "probe_stderr": stderr,
        "probe_error": error,
        "version_banner": [line.rstrip() for line in banner.splitlines() if line.strip()],
        "gpu_build_tokens_searched": list(GPU_BUILD_TOKENS),
        "gpu_build_tokens_found": found,
        "requested_backend": "gpu_if_available",
        "effective_backend": "cpu",
        "fallback": True,
        "reason": (
            "SIESTA --version reports no GPU/CUDA/OpenACC/ROCm/MAGMA support in this build; "
            "no scientifically equivalent GPU backend exists for this SCF, so CPU is the "
            "effective backend"
            if not found and returncode == 0
            else f"probe inconclusive (returncode={returncode}, tokens={found}, error={error!r}); "
            "CPU used and the inconclusive probe recorded"
        ),
    }


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    branch: str
    positions_ang: list[list[float]]
    direction: fdp.Direction | None = None
    sign: int = 0
    delta_ang: float | None = None

    @property
    def sign_label(self) -> str:
        return {1: "plus", -1: "minus"}.get(self.sign, "")

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "branch": self.branch,
            "delta_ang": self.delta_ang,
            "sign": self.sign,
            "sign_label": self.sign_label,
            "positions_ang": [list(map(float, position)) for position in self.positions_ang],
        }
        if self.direction is not None:
            payload.update(self.direction.to_metadata())
        return payload


def _delta_tag(delta_ang: float) -> str:
    return "d" + f"{delta_ang:.6g}".replace(".", "p").replace("-", "m")


def plan_runs(
    positions_ang: list[list[float]],
    *,
    directions: list[fdp.Direction],
    delta_ang_values: list[float],
    method: str = "central",
) -> list[RunSpec]:
    """Every SIESTA run this campaign needs, in a deterministic order."""
    runs = [RunSpec(run_id=BRANCH_EQUILIBRIUM, branch=BRANCH_EQUILIBRIUM, positions_ang=positions_ang)]
    for delta in delta_ang_values:
        runs.append(
            RunSpec(
                run_id=f"{BRANCH_FC}_{_delta_tag(delta)}",
                branch=BRANCH_FC,
                positions_ang=positions_ang,
                delta_ang=float(delta),
            )
        )
    for direction in directions:
        for delta in delta_ang_values:
            for sign in fdp.signs_for_method(method):
                label = "plus" if sign > 0 else "minus"
                runs.append(
                    RunSpec(
                        run_id=f"fd_{direction.name}_{_delta_tag(delta)}_{label}",
                        branch=BRANCH_FD,
                        positions_ang=fdp.displace(positions_ang, direction, sign * float(delta)),
                        direction=direction,
                        sign=sign,
                        delta_ang=float(delta),
                    )
                )
    return runs


def topology_report(
    positions_ang: list[list[float]],
    *,
    directions: list[fdp.Direction],
    delta_ang_values: list[float],
    cutoff_ang: list[float],
    lattice_vectors_ang: list[list[float]],
    method: str,
) -> list[dict[str, Any]]:
    """Neighbour-list margin of the largest excursion of every direction."""
    largest = max(float(delta) for delta in delta_ang_values)
    rows = []
    for direction in directions:
        margin = fdp.topology_margin(
            positions_ang,
            direction=direction,
            delta_ang=largest,
            cutoff_ang=cutoff_ang,
            lattice_vectors_ang=lattice_vectors_ang,
            method=method,
        )
        rows.append({"direction_name": direction.name, "delta_ang": largest, **margin.to_dict()})
    return rows


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _artifact_hashes(run_dir: Path, label: str) -> dict[str, Any]:
    """sha256 of everything downstream may read, plus what the FC branch adds."""
    named = {
        "run_fdf": run_dir / "RUN.fdf",
        "run_out": run_dir / "RUN.out",
        "tshs": run_dir / f"{label}.TSHS",
        "orb_indx": run_dir / f"{label}.ORB_INDX",
        "fc": run_dir / f"{label}.FC",
        "struct_out": run_dir / f"{label}.STRUCT_OUT",
    }
    hashes = {name: file_sha256(path) for name, path in named.items() if path.is_file()}
    dhsdr = sorted(run_dir.glob("*dHSdR.nc"))
    if dhsdr:
        hashes["dhsdr"] = file_sha256(dhsdr[0])
    # SIESTA's FC run writes the H and S of each of its own +-displacements;
    # C09 compares them against dHSdR.nc without a second SCF.
    displaced = {path.name: file_sha256(path) for path in sorted(run_dir.glob(f"{label}.[0-9]*.TSHS"))}
    if displaced:
        hashes["fc_displacement_tshs"] = displaced
    for suffix in ("VT", "VH"):
        path = run_dir / f"{label}.{suffix}"
        if path.is_file():
            hashes[suffix.lower()] = file_sha256(path)
    return hashes


def fermi_level_ev(tshs: Path) -> float | None:
    """The energy zero stored in a TSHS, or ``None`` if it cannot be read.

    Recorded because the two sides of GO-2 do not share it: sisl hands out the
    TSHS Hamiltonian already shifted, ``H - E_F S``, while ``dHSdR.nc`` holds
    the unshifted ``dH/dR``. A finite difference of the shifted matrix is
    ``dH - (dE_F) S - E_F dS``, and ``E_F`` moves with the displacement, so C09
    needs this number for every member of the pair, not once for the campaign.
    """
    try:
        import sisl

        return float(sisl.get_sile(str(tshs)).read_fermi_level())
    except Exception:  # noqa: BLE001 - a missing energy zero is reported, never guessed
        return None


def _directory_bytes(run_dir: Path) -> int:
    return sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())


def _required_artifacts(branch: str, run_dir: Path, label: str) -> list[str]:
    """Missing artifacts for this branch, by name. Empty means complete."""
    required = {
        BRANCH_FC: [f"{label}.FC", f"{label}.ORB_INDX"],
    }.get(branch, [f"{label}.TSHS", f"{label}.ORB_INDX"])
    missing = [name for name in required if not (run_dir / name).is_file()]
    if branch == BRANCH_FC and not sorted(run_dir.glob("*dHSdR.nc")):
        missing.append("*dHSdR.nc")
    return missing


def stage_run(
    spec: RunSpec,
    *,
    run_dir: Path,
    base_fdf: Path,
    pseudo_dir: Path,
    atom_count: int,
    fc_tolerances: dict[str, str],
) -> dict[str, Any]:
    """Write RUN.fdf and the pseudopotentials of one run into a clean directory."""
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    materialized = materialize_sample_fdf(
        base_fdf,
        run_dir / "RUN.fdf",
        positions_ang=spec.positions_ang,
        system_label=spec.run_id,
        system_name=spec.run_id,
        # Even the FC branch is materialised single-point first: it strips the
        # inherited MD layer, and write_fc_layer then appends the FC one.
        single_point=True,
    )
    if spec.branch == BRANCH_FC:
        write_fc_layer(
            run_dir / "RUN.fdf",
            displacement_ang=float(spec.delta_ang),
            atom_count=atom_count,
            tolerances=fc_tolerances,
        )
    pseudos = sorted(path for path in pseudo_dir.iterdir() if path.suffix in {".psf", ".psml", ".vps"})
    if not pseudos:
        raise SiestaReferenceError(f"No pseudopotentials in {pseudo_dir}.")
    for source in pseudos:
        shutil.copy2(source, run_dir / source.name)
    return {
        **materialized.metadata,
        "pseudopotential_sha256": {source.name: file_sha256(source) for source in pseudos},
        "physics_fingerprint": physics_fingerprint((run_dir / "RUN.fdf").read_text(encoding="utf-8")),
    }


def run_signature(spec: RunSpec, staged: dict[str, Any], shared: dict[str, Any]) -> str:
    """Signature of one run's inputs.

    The FC thresholds are part of the FC branch only: they change what
    ``dHSdR.nc`` contains and nothing about a single-point SCF, so re-running
    the sparsification sweep must not throw away the H and S of the finite
    differences.
    """
    payload = {
        "schema": SCHEMA,
        "run": spec.to_metadata(),
        "materialized_fdf_sha256": staged["materialized_fdf_sha256"],
        "physics_fingerprint": staged["physics_fingerprint"],
        "pseudopotential_sha256": staged["pseudopotential_sha256"],
        **shared,
    }
    if spec.branch != BRANCH_FC:
        payload.pop("fc_tolerances", None)
    return input_signature_sha256(payload)


def execute_run(spec: RunSpec, *, run_dir: Path, siesta_command: str) -> dict[str, Any]:
    """One SIESTA invocation, with its wall time and the children RSS watermark."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.time()
    record = run_siesta(run_dir, command=siesta_command, use_shell=False)
    wall = time.time() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {
        "command": " ".join(record["command"]) if isinstance(record["command"], list) else record["command"],
        "returncode": int(record["returncode"]),
        "wall_seconds": round(wall, 3),
        # ponytail: RUSAGE_CHILDREN.ru_maxrss is a high-water mark over all
        # children of this process, so it only rises. Honest per run because the
        # campaign is sequential; parallelising it would need per-run cgroups.
        "peak_rss_children_kb": int(after),
        "peak_rss_children_rose_kb": int(max(0, after - before)),
    }


def evaluate_run(
    spec: RunSpec,
    *,
    run_dir: Path,
    label: str,
    equilibrium_orb_indx: str | None,
) -> dict[str, Any]:
    """Everything that decides whether this run may be certified."""
    status = parse_siesta_output(run_dir / "RUN.out", run_dir / "RUN.fdf")
    hashes = _artifact_hashes(run_dir, label)
    missing = _required_artifacts(spec.branch, run_dir, label)
    problems: list[str] = []
    if not status.get("job_completed"):
        problems.append("job_not_completed")
    if not status.get("scf_converged"):
        problems.append("scf_not_converged")
    if status.get("explicit_nonconvergence"):
        problems.append("explicit_nonconvergence")
    if missing:
        problems.append("missing_artifacts:" + ",".join(missing))
    if equilibrium_orb_indx is not None and hashes.get("orb_indx") != equilibrium_orb_indx:
        # Same base fdf, same species, same basis: the orbital ordering cannot
        # legitimately change with a displacement. If it did, the matrix rows of
        # this run do not mean what the equilibrium rows mean.
        problems.append("orbital_mapping_differs_from_equilibrium")
    if spec.branch != BRANCH_FC and not missing:
        # The FC branch is exempt: SIESTA itself moves the atoms, so its final
        # geometry is a displaced one by construction.
        geometry_error = reference_output_geometry_error(run_dir, run_dir / f"{label}.TSHS")
        if geometry_error:
            problems.append(geometry_error)
    return {
        "siesta_output_status": status,
        "fermi_level_ev": (
            fermi_level_ev(run_dir / f"{label}.TSHS")
            if spec.branch != BRANCH_FC and (run_dir / f"{label}.TSHS").is_file()
            else None
        ),
        "artifact_sha256": hashes,
        "missing_artifacts": missing,
        "problems": problems,
        "certified": not problems,
        "disk_bytes": _directory_bytes(run_dir),
    }


def process_run(
    spec: RunSpec,
    *,
    output_root: Path,
    base_fdf: Path,
    pseudo_dir: Path,
    atom_count: int,
    fc_tolerances: dict[str, str],
    siesta_command: str,
    shared_signature_inputs: dict[str, Any],
    equilibrium_orb_indx: str | None,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / spec.run_id
    record_path = run_dir / "run_record.json"
    previous = _read_json(record_path) if record_path.is_file() else None

    if previous and not overwrite:
        # Reuse needs the same inputs *and* intact outputs; a certified record
        # whose artifacts vanished is not a ground state we still own.
        label = str(previous.get("system_label") or spec.run_id)
        intact = previous.get("certified") and not _required_artifacts(spec.branch, run_dir, label)
        if intact:
            probe = stage_probe_signature(
                spec,
                base_fdf=base_fdf,
                pseudo_dir=pseudo_dir,
                atom_count=atom_count,
                fc_tolerances=fc_tolerances,
                shared_signature_inputs=shared_signature_inputs,
                run_dir=run_dir,
            )
            if probe == previous.get("input_signature_sha256"):
                return {**previous, "status": "reused"}

    if dry_run:
        return {
            **spec.to_metadata(),
            "status": "planned",
            "run_dir": str(run_dir),
            "certified": False,
        }

    staged = stage_run(
        spec,
        run_dir=run_dir,
        base_fdf=base_fdf,
        pseudo_dir=pseudo_dir,
        atom_count=atom_count,
        fc_tolerances=fc_tolerances,
    )
    signature = run_signature(spec, staged, shared_signature_inputs)
    label = read_system_label(run_dir / "RUN.fdf")
    started_at = datetime.now(timezone.utc).isoformat()
    execution = execute_run(spec, run_dir=run_dir, siesta_command=siesta_command)
    evaluation = evaluate_run(
        spec, run_dir=run_dir, label=label, equilibrium_orb_indx=equilibrium_orb_indx
    )
    if execution["returncode"] != 0:
        evaluation["problems"].append(f"siesta_returncode_{execution['returncode']}")
        evaluation["certified"] = False
    row = {
        **spec.to_metadata(),
        "status": "ok" if evaluation["certified"] else "failed",
        "run_dir": str(run_dir),
        "system_label": label,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input_signature_sha256": signature,
        "staged": staged,
        **execution,
        **evaluation,
    }
    _write_json(record_path, row)
    return row


def stage_probe_signature(
    spec: RunSpec,
    *,
    base_fdf: Path,
    pseudo_dir: Path,
    atom_count: int,
    fc_tolerances: dict[str, str],
    shared_signature_inputs: dict[str, Any],
    run_dir: Path,
) -> str:
    """Signature the run *would* get now, computed without touching its outputs."""
    probe_dir = run_dir.parent / f".probe_{run_dir.name}"
    try:
        staged = stage_run(
            spec,
            run_dir=probe_dir,
            base_fdf=base_fdf,
            pseudo_dir=pseudo_dir,
            atom_count=atom_count,
            fc_tolerances=fc_tolerances,
        )
        return run_signature(spec, staged, shared_signature_inputs)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Campaign
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def certification_set(rows: list[dict[str, Any]], *, delta_ang_values: list[float]) -> dict[str, Any]:
    """The pairs C09 is allowed to differentiate, and the ones it is not.

    A ``+-`` pair is usable only if *both* members are certified; an FC run only
    if its own dHSdR/FC artifacts survived certification. Nothing that failed
    SCF convergence can appear here.
    """
    certified = {row["run_id"]: row for row in rows if row.get("certified")}
    fc_by_delta = {
        float(row["delta_ang"]): row["run_id"]
        for row in rows
        if row.get("branch") == BRANCH_FC and row.get("certified")
    }
    pairs: list[dict[str, Any]] = []
    for row in rows:
        if row.get("branch") != BRANCH_FD or row.get("sign") != 1:
            continue
        minus_id = row["run_id"].rsplit("_", 1)[0] + "_minus"
        pairs.append(
            {
                "direction_name": row.get("direction_name"),
                "direction_kind": row.get("direction_kind"),
                "direction_hash": row.get("direction_hash"),
                "delta_ang": row.get("delta_ang"),
                "plus_run_id": row["run_id"],
                "minus_run_id": minus_id,
                "plus_certified": row["run_id"] in certified,
                "minus_certified": minus_id in certified,
                "ready_for_central_difference": row["run_id"] in certified and minus_id in certified,
            }
        )
    per_delta = []
    for delta in delta_ang_values:
        ready = [pair for pair in pairs if pair["delta_ang"] == delta and pair["ready_for_central_difference"]]
        per_delta.append(
            {
                "delta_ang": delta,
                "fc_run_id": fc_by_delta.get(delta),
                "fc_dhs_available": delta in fc_by_delta,
                "ready_fd_pair_count": len(ready),
                "ready_fd_directions": [pair["direction_name"] for pair in ready],
                # GO-2 compares dHSdR.nc against an explicit FD of the same H/S
                # at the same amplitude: both sides must exist for this delta.
                "fc_vs_fd_comparable": bool(delta in fc_by_delta and ready),
            }
        )
    return {
        "equilibrium_run_id": BRANCH_EQUILIBRIUM if BRANCH_EQUILIBRIUM in certified else None,
        "fd_pairs": pairs,
        "per_delta": per_delta,
        "uncertified_run_ids": sorted(row["run_id"] for row in rows if not row.get("certified")),
        "all_deltas_comparable": bool(per_delta) and all(entry["fc_vs_fd_comparable"] for entry in per_delta),
    }


def run_campaign(
    *,
    material_fdf: Path = DEFAULT_MATERIAL_FDF,
    basis_dir: Path = DEFAULT_BASIS_DIR,
    pseudo_dir: Path = DEFAULT_PSEUDO_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    siesta_command: str = DEFAULT_SIESTA_COMMAND,
    delta_ang_values: list[float] | None = None,
    method: str = "central",
    seed: int = 0,
    directions: list[fdp.Direction] | None = None,
    fc_tolerances: dict[str, str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_topology_change: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    fc_tolerances = fc_tolerances or {"dhdr": DEFAULT_DHDR_TOLERANCE, "dsdr": DEFAULT_DSDR_TOLERANCE}
    deltas = sorted(float(value) for value in (delta_ang_values or fdp.delta_sweep()))

    base_fdf = canonical_ang_base_fdf(material_fdf, output_root / "inputs" / "base_ang.fdf")
    structure = extract_fdf_structure(base_fdf)
    positions = [list(position) for position in structure.positions_ang]
    lattice = [list(vector) for vector in structure.lattice_vectors_ang]

    contract = build_orbital_contract(material_fdf.parent.name, base_fdf, basis_dir)
    # The default set is the GO-2 perturbation space. A caller may substitute its
    # own directions (C20 displaces along the pre-registered E2g members, which
    # are frozen by value in the protocol and must not be re-derived here); the
    # manifest records whichever set was used, so the campaign stays self-describing.
    direction_source_default = directions is None
    if directions is None:
        directions = fdp.default_direction_set(structure.atom_count, seed=seed)
    for direction in directions:
        if direction.atom_count != structure.atom_count:
            raise SiestaReferenceError(
                f"direction {direction.name!r} moves {direction.atom_count} atoms, the structure "
                f"has {structure.atom_count}"
            )
    topology = topology_report(
        positions,
        directions=directions,
        delta_ang_values=deltas,
        cutoff_ang=fdp.atom_cutoff_radii_ang(contract),
        lattice_vectors_ang=lattice,
        method=method,
    )
    crossing = [row for row in topology if not row["topology_preserved"]]
    if crossing and not allow_topology_change:
        raise SiestaReferenceError(
            "Perturbation changes the neighbour topology for "
            f"{[row['direction_name'] for row in crossing]}; a derivative across a graph "
            "discontinuity is not a derivative. Re-run with a smaller delta, or with "
            "--allow-topology-change if the campaign deliberately measures the crossing."
        )

    preflight = backend_preflight(siesta_command)
    runtime = collect_siesta_runtime(siesta_command.split()[0])
    runtime_ref = write_siesta_runtime_record(runtime, SIESTA_RUNTIME_DIR)
    runtime_validation = validate_siesta_runtime_ref(runtime_ref, effective_executable=siesta_command.split()[0])

    shared_signature_inputs = {
        "base_fdf_sha256": file_sha256(base_fdf),
        "siesta_binary_sha256": runtime_ref.get("binary_sha256"),
        "siesta_runtime_record_sha256": runtime_ref.get("record_sha256"),
        "fc_tolerances": dict(fc_tolerances),
        "method": method,
    }

    specs = plan_runs(positions, directions=directions, delta_ang_values=deltas, method=method)
    rows: list[dict[str, Any]] = []
    equilibrium_orb_indx: str | None = None
    for spec in specs:
        row = process_run(
            spec,
            output_root=output_root,
            base_fdf=base_fdf,
            pseudo_dir=pseudo_dir,
            atom_count=structure.atom_count,
            fc_tolerances=fc_tolerances,
            siesta_command=siesta_command,
            shared_signature_inputs=shared_signature_inputs,
            equilibrium_orb_indx=equilibrium_orb_indx,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        if spec.branch == BRANCH_EQUILIBRIUM:
            equilibrium_orb_indx = (row.get("artifact_sha256") or {}).get("orb_indx")
            if row.get("certified") is False and not dry_run:
                raise SiestaReferenceError(
                    f"The equilibrium run is not certifiable ({row.get('problems')}); "
                    "nothing downstream can be compared against it."
                )
        rows.append(row)
        print(
            f"[EPC-SIESTA-REF] {row['status']:>8}  {spec.run_id}  "
            f"{row.get('wall_seconds', '')}s  {'' if row.get('certified') else row.get('problems', '')}"
        )

    fingerprints = sorted({(row.get("staged") or {}).get("physics_fingerprint") for row in rows} - {None})
    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "material_fdf": str(material_fdf),
        "material_fdf_sha256": file_sha256(material_fdf),
        "base_ang_fdf": str(base_fdf),
        "base_ang_fdf_sha256": file_sha256(base_fdf),
        "basis_dir": str(basis_dir),
        "pseudo_dir": str(pseudo_dir),
        "output_root": str(output_root),
        "siesta_command": siesta_command,
        "siesta_runtime_ref": runtime_ref,
        "siesta_runtime_validation": runtime_validation,
        "compute_policy": {
            **{key: preflight[key] for key in ("requested_backend", "effective_backend", "fallback", "reason")},
            "device": "cpu",
            "precision": {
                "value": None,
                "reason": "the SIESTA binary does not report its floating-point build precision",
            },
            "preflight": preflight,
        },
        "orbital_contract": {
            "orbital_contract_hash": contract.get("orbital_contract_hash"),
            "basis_contract_hash": contract.get("basis_contract_hash"),
            "atom_count": contract.get("atom_count"),
            "orbital_count": contract.get("orbital_count"),
            "declared_unpopulated_species": contract.get("declared_unpopulated_species"),
        },
        "perturbation_space": {
            "normalization": fdp.NORMALIZATION,
            "method": method,
            "seed": seed,
            "direction_source": "default_direction_set" if direction_source_default else "caller",
            "directions": [direction.to_dict() for direction in directions],
            **fdp.delta_sweep_status(deltas),
        },
        "fc": {
            "save_dhs": True,
            "first_atom": 1,
            "last_atom": structure.atom_count,
            "displacement_ang_values": deltas,
            "displacement_equals_fd_delta": True,
            "tolerances": dict(fc_tolerances),
            "sparsification": "off_by_default_zero_tolerances",
        },
        "topology_margin": topology,
        "topology_preserved": not crossing,
        "physics_fingerprints": fingerprints,
        "physics_identical_across_branches": len(fingerprints) <= 1,
        "runs_total": len(rows),
        "runs_certified": len([row for row in rows if row.get("certified")]),
        "runs_reused": len([row for row in rows if row.get("status") == "reused"]),
        "runs_failed": len([row for row in rows if row.get("status") == "failed"]),
        "resources": {
            "wall_seconds_total": round(sum(float(row.get("wall_seconds") or 0.0) for row in rows), 3),
            "peak_rss_children_kb": max([int(row.get("peak_rss_children_kb") or 0) for row in rows] or [0]),
            "disk_bytes_total": sum(int(row.get("disk_bytes") or 0) for row in rows),
        },
        "certification_set": certification_set(rows, delta_ang_values=deltas),
        "rows": rows,
    }
    if not dry_run and not manifest["physics_identical_across_branches"]:
        raise SiestaReferenceError(
            f"Branches did not run the same physics: {len(fingerprints)} distinct fdf fingerprints."
        )
    _write_json(output_root / "epc_siesta_reference_manifest.json", manifest)
    _write_status_csv(output_root / "epc_siesta_reference_status.csv", rows)
    return manifest


def _write_status_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            status = row.get("siesta_output_status") or {}
            writer.writerow(
                {
                    **row,
                    "direction": row.get("direction_name") or "",
                    "scf_converged": status.get("scf_converged"),
                    "job_completed": status.get("job_completed"),
                    "error": ";".join(row.get("problems") or []),
                }
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material-fdf", type=Path, default=DEFAULT_MATERIAL_FDF)
    parser.add_argument("--basis-dir", type=Path, default=DEFAULT_BASIS_DIR)
    parser.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--siesta-command", default=DEFAULT_SIESTA_COMMAND)
    parser.add_argument(
        "--delta-ang",
        default="",
        help="Comma separated amplitudes in Ang. Default: the geometric sweep of fd_perturbation_space.",
    )
    parser.add_argument("--method", default="central", choices=sorted(fdp.VALID_METHODS))
    parser.add_argument("--seed", type=int, default=0, help="Seed of the collective direction.")
    parser.add_argument("--fc-dhdr-tolerance", default=DEFAULT_DHDR_TOLERANCE)
    parser.add_argument("--fc-dsdr-tolerance", default=DEFAULT_DSDR_TOLERANCE)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even when the signature matches.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight and plan only; no SCF.")
    parser.add_argument("--allow-topology-change", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    deltas = [float(value) for value in args.delta_ang.split(",") if value.strip()] or None
    try:
        manifest = run_campaign(
            material_fdf=args.material_fdf,
            basis_dir=args.basis_dir,
            pseudo_dir=args.pseudo_dir,
            output_root=args.output_root,
            siesta_command=args.siesta_command,
            delta_ang_values=deltas,
            method=args.method,
            seed=args.seed,
            fc_tolerances={"dhdr": args.fc_dhdr_tolerance, "dsdr": args.fc_dsdr_tolerance},
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            allow_topology_change=args.allow_topology_change,
        )
    except (SiestaReferenceError, fdp.FdPerturbationError) as exc:
        print(f"[EPC-SIESTA-REF][ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "runs_total",
                    "runs_certified",
                    "runs_reused",
                    "runs_failed",
                    "physics_identical_across_branches",
                )
            }
            | {"all_deltas_comparable": manifest["certification_set"]["all_deltas_comparable"]},
            sort_keys=True,
        )
    )
    return 0 if manifest["runs_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
