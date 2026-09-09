#!/usr/bin/env python3
"""Archive SIESTA FC output with provenance and optionally run VIBRA (C07 / E-F_001-S9).

An FC run is the only producer of the two artifacts the EPC reference path needs:
the force constants (``*.FC``) and the matrix derivatives (``*.dHSdR.nc``). This
script archived them by name only — ``dHSdR.nc`` was not even in the copy list —
so a rerun could silently drop the derivative file and nothing recorded which
geometry, basis, displacement or SIESTA binary produced any of it.

This is still only an archiver plus a VIBRA post-process. What it adds is the
provenance every downstream EPC artifact has to cite:

* the copied bytes, hashed at the source and rechecked in the archive;
* the FC physics — ``FC.Displacement``, the ``FC.First``/``FC.Last`` atom range
  and the ``FC.dHdR``/``FC.dSdR`` thresholds — read from the RUN.fdf and, when
  ``dHSdR.nc`` can be decoded, cross-checked against what SIESTA wrote inside it;
* the geometry/basis signature the whole archive belongs to;
* the C01 SIESTA runtime record, compared with the version banner of the run
  output that actually produced these files.

Two rules keep reruns safe: nothing here ever deletes an artifact, and an
archive whose geometry/basis signature differs from the one already recorded is
refused instead of overwritten, so a manifest can never link artifacts of two
different structures.

Statuses are the C02 vocabulary (``PRESENT_VALID`` / ``PRESENT_UNVERIFIED`` /
``MISSING`` / ``INVALID``) produced by ``run_inventory.classify_epc_artifact``:
a file sitting in the archive that no producer declares stays
``PRESENT_UNVERIFIED``, never ``PRESENT_VALID``.

No PAO-projected couplingate lives here: bytes and declarations only. GO-2 (C09) still has to
certify the derivatives against explicit finite differences; the archived ``.FC``
and VIBRA output become normalised modes only through the C17 phonon contract,
``Comparison/scripts/phonon_provider.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _directory in (REPO_ROOT / "shared", SCRIPT_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from benchmark_manifest import (  # noqa: E402
    canonical_sha256,
    extract_siesta_version_from_text,
    file_sha256,
)
from fdf_materialization import BOHR_TO_ANG  # noqa: E402
from orbital_contract import build_orbital_contract, fdf_key  # noqa: E402
from reference_provenance import (  # noqa: E402
    geometry_cell_species_sha256,
    system_label_from_fdf,
)
from run_inventory import (  # noqa: E402
    INVALID,
    MISSING,
    PRESENT_UNVERIFIED,
    PRESENT_VALID,
    classify_epc_artifact,
    epc_siesta_runtime_block,
    evidence_item,
    worst_status,
)

ARCHIVE_SCHEMA = "phonon_archive_manifest_v1"
MANIFEST_NAME = "phonon_manifest.json"

# SIESTA writes lengths in Bohr unless the directive carries its own unit.
LENGTH_UNITS_TO_BOHR = {"bohr": 1.0, "ang": 1.0 / BOHR_TO_ANG, "angstrom": 1.0 / BOHR_TO_ANG}
DEFAULT_LENGTH_UNIT = "bohr"

BASIS_PATTERNS = ("*.ion.xml", "*.ion", "*.psf", "*.psml")

ARCHIVE_LIMITATIONS = (
    "PRESENT_VALID means the archived bytes match what the producer declared; it says "
    "nothing about whether the derivatives are physically correct (GO-2 is still open)",
    "VIBRA output is archived, not interpreted: frequency, eigenvector, mass "
    "normalisation and ASR residual come from phonon_provider.py (C17), not from here",
)


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def artifact_kind(name: str, origin: str) -> str:
    """Which EPC artifact one archived file is, from its SIESTA name."""
    if name.endswith(".FC"):
        return "force_constants"
    if "dHSdR" in name:
        return "dhsdr"
    if name.endswith(".ORB_INDX"):
        return "orb_indx"
    if name.endswith((".ion.xml", ".ion")):
        return "basis"
    return "vibra_output" if origin == "produced_by_vibra" else "fc_run_output"


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------


def copy_outputs(fc_run_dir: Path, output_dir: Path) -> dict[str, dict[str, Any]]:
    """Copy the FC run artifacts, skipping the ones already archived byte-identical.

    Returns one record per archived file: its source, the sha256 hashed *at the
    source*, and whether this run had to write it. Skipping identical files is
    what makes a rerun idempotent down to the mtimes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("*.FC", "*.fdf", "*.out", "*.xml", "*.xyz", "*.FA", "*.XV", "*dHSdR.nc", "*.ORB_INDX")
    copied: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for src in sorted(fc_run_dir.glob(pattern)):
            if not src.is_file() or src.name in copied:
                continue
            dst = output_dir / src.name
            source_sha256 = file_sha256(src)
            unchanged = dst.is_file() and file_sha256(dst) == source_sha256
            if not unchanged:
                shutil.copy2(src, dst)
            copied[src.name] = {
                "path": str(dst),
                "source": _relative(src),
                "source_sha256": source_sha256,
                "action": "unchanged" if unchanged else "copied",
            }
    return copied


# ---------------------------------------------------------------------------
# FC physics declared by the RUN.fdf
# ---------------------------------------------------------------------------


def fdf_directives(fdf_path: Path) -> dict[str, str]:
    """``key -> rest of the line`` for every FDF directive.

    ``orbital_contract._fdf_value`` returns only the first token, which drops the
    unit of ``FC.Displacement 0.04 Bohr``; the FC physics needs the unit, so the
    whole value is kept here and normalised with the same label rule.
    """
    directives: dict[str, str] = {}
    for raw in fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            directives.setdefault(fdf_key(parts[0]), parts[1].strip())
    return directives


def _parse_physical(raw: str | None, units_to_canonical: dict[str, float] | None) -> dict[str, Any]:
    """One FDF physical value: number, unit token, and the converted value."""
    record: dict[str, Any] = {"raw": raw, "value": None, "unit": None, "unit_assumed": False}
    if raw is None:
        return record
    parts = raw.split()
    try:
        record["value"] = float(parts[0])
    except (IndexError, ValueError):
        record["error"] = "value_is_not_a_number"
        return record
    if len(parts) > 1:
        record["unit"] = parts[1]
    if units_to_canonical is None:
        return record
    unit = (record["unit"] or DEFAULT_LENGTH_UNIT).lower()
    record["unit_assumed"] = record["unit"] is None
    factor = units_to_canonical.get(unit)
    if factor is None:
        record["error"] = f"unknown_length_unit:{record['unit']}"
        return record
    record["bohr"] = record["value"] * factor
    record["ang"] = record["bohr"] * BOHR_TO_ANG
    return record


def fc_parameters(fdf_path: Path) -> dict[str, Any]:
    """The FC physics an EPC artifact must cite, as the RUN.fdf declares it."""
    directives = fdf_directives(fdf_path)

    def value(key: str) -> str | None:
        return directives.get(fdf_key(key))

    def integer(key: str) -> int | None:
        raw = value(key)
        try:
            return int(str(raw).split()[0])
        except (AttributeError, IndexError, ValueError):
            return None

    first_atom = integer("FC.First")
    last_atom = integer("FC.Last")
    return {
        "run_fdf": _relative(fdf_path),
        "run_fdf_sha256": file_sha256(fdf_path),
        "md_type_of_run": value("MD.TypeOfRun"),
        "save_dhs": value("FC.Save.dHS"),
        "displacement": _parse_physical(value("FC.Displacement"), LENGTH_UNITS_TO_BOHR),
        "atom_range": {
            "first_atom": first_atom,
            "last_atom": last_atom,
            "atoms": list(range(first_atom, last_atom + 1))
            if first_atom is not None and last_atom is not None
            else None,
        },
        "thresholds": {
            "dHdR": _parse_physical(value("FC.dHdR.Tolerance"), None),
            "dSdR": _parse_physical(value("FC.dSdR.Tolerance"), None),
        },
        "declared_by": "RUN.fdf directives (SIESTA reads these, not metadata.json)",
    }


# ---------------------------------------------------------------------------
# What the dHSdR.nc itself states
# ---------------------------------------------------------------------------


def _close(left: Any, right: Any, *, rel_tol: float = 1e-6) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=1e-12)
    )


def dhsdr_cross_checks(metadata: dict[str, Any], fc: dict[str, Any]) -> list[dict[str, Any]]:
    """The file and the FDF must describe the same displacement, atoms, thresholds."""
    checks: list[dict[str, Any]] = []
    declared = fc["displacement"].get("bohr")
    observed = metadata["displacement"]["bohr"]
    checks.append(
        evidence_item(
            "displacement_matches_fdf",
            "absent" if declared is None else ("pass" if _close(declared, observed) else "fail"),
            f"fdf={fc['displacement']['raw']!r} -> {declared} Bohr, dHSdR.nc={observed} Bohr",
        )
    )

    atoms = fc["atom_range"]["atoms"]
    written = list(metadata["displaced_atoms"])
    checks.append(
        evidence_item(
            "displaced_atom_range_matches_fdf",
            "absent" if atoms is None else ("pass" if written == atoms else "fail"),
            f"FC.First/FC.Last -> {atoms}, dHSdR.nc atom_index = {written}",
        )
    )

    for key, declared_value, observed_value in (
        ("dHdR", fc["thresholds"]["dHdR"]["value"], metadata["thresholds"]["dHdR"]["value_ry_bohr"]),
        ("dSdR", fc["thresholds"]["dSdR"]["value"], metadata["thresholds"]["dSdR"]["value_inv_bohr"]),
    ):
        checks.append(
            evidence_item(
                f"{key}_threshold_matches_fdf",
                "absent"
                if declared_value is None
                else ("pass" if _close(declared_value, observed_value) else "fail"),
                f"fdf={declared_value}, dHSdR.nc={observed_value}",
            )
        )
    return checks


FDF_TRUE = {"t", "true", ".true.", "yes", "1"}


def dhsdr_block(path: Path | None, fc: dict[str, Any]) -> dict[str, Any]:
    """Decode the archived ``dHSdR.nc`` and confront it with the FDF."""
    if path is None:
        # Only a run that asked for the derivatives is missing something.
        expected = str(fc.get("save_dhs") or "").strip().lower() in FDF_TRUE
        return {
            "status": MISSING,
            "expected": expected,
            "detail": "FC.Save.dHS is on but no *dHSdR.nc reached the archive"
            if expected
            else "FC.Save.dHS is off in the RUN.fdf; this run produces no matrix derivatives",
        }
    try:
        from read_siesta_dhsdr import read_dhsdr  # lazy: only this block needs netCDF4
    except ImportError as exc:  # noqa: BLE001 - an undecoded file is unverified, not invalid
        return {
            "status": PRESENT_UNVERIFIED,
            "path": _relative(path),
            "detail": f"netCDF4 unavailable, the file could not be decoded: {exc!r}",
        }
    try:
        decoded = read_dhsdr(path)
    except Exception as exc:  # noqa: BLE001 - a file that is not the schema is evidence
        return {"status": INVALID, "path": _relative(path), "detail": repr(exc)}

    metadata = decoded.metadata()
    checks = dhsdr_cross_checks(metadata, fc)
    if any(item["outcome"] == "fail" for item in checks) or decoded.status == INVALID:
        status = INVALID
    elif any(item["outcome"] == "absent" for item in checks):
        status = PRESENT_UNVERIFIED
    else:
        status = PRESENT_VALID
    return {
        "status": status,
        "path": _relative(path),
        "reader": metadata,
        "cross_checks": checks,
    }


# ---------------------------------------------------------------------------
# Geometry / basis the archive belongs to
# ---------------------------------------------------------------------------


def resolve_run_fdf(fc_run_dir: Path, run_fdf: Path | None = None) -> Path:
    if run_fdf is not None:
        if not run_fdf.is_file():
            raise RuntimeError(f"No existe el RUN.fdf indicado: {run_fdf}")
        return run_fdf
    default = fc_run_dir / "RUN.fdf"
    if default.is_file():
        return default
    candidates = sorted(path for path in fc_run_dir.glob("*.fdf") if path.is_file())
    if len(candidates) != 1:
        raise RuntimeError(
            f"No se pudo determinar el fdf del run FC en {fc_run_dir}: {[p.name for p in candidates]}; "
            "usa --run-fdf"
        )
    return candidates[0]


def geometry_basis_block(run_fdf: Path, basis_dir: Path) -> dict[str, Any]:
    """One signature for the geometry *and* the basis every artifact belongs to."""
    basis_files = {
        path.name: file_sha256(path)
        for pattern in BASIS_PATTERNS
        for path in sorted(basis_dir.glob(pattern))
        if path.is_file()
    }
    label = system_label_from_fdf(run_fdf)
    try:
        contract = build_orbital_contract(label or run_fdf.stem, run_fdf, basis_dir)
        orbital_contract = {
            "basis_dir": _relative(basis_dir),
            "basis_contract_hash": contract["basis_contract_hash"],
            "orbital_contract_hash": contract["orbital_contract_hash"],
            "atom_count": contract["atom_count"],
            "orbital_count": contract["orbital_count"],
            "problems": contract["problems"],
        }
    except Exception as exc:  # noqa: BLE001 - a missing .ion.xml is unverified, not fatal
        orbital_contract = {
            "basis_dir": _relative(basis_dir),
            "basis_contract_hash": None,
            "orbital_contract_hash": None,
            "unavailable_reason": repr(exc),
        }

    block = {
        "run_fdf": _relative(run_fdf),
        "run_fdf_sha256": file_sha256(run_fdf),
        "system_label": label,
        "geometry_cell_species_sha256": geometry_cell_species_sha256(run_fdf),
        "basis_files": dict(sorted(basis_files.items())),
        "orbital_contract": orbital_contract,
    }
    block["signature_sha256"] = canonical_sha256(
        {
            "geometry_cell_species_sha256": block["geometry_cell_species_sha256"],
            "basis_files": block["basis_files"],
            "basis_contract_hash": orbital_contract["basis_contract_hash"],
            "orbital_contract_hash": orbital_contract["orbital_contract_hash"],
        }
    )
    return block


# ---------------------------------------------------------------------------
# SIESTA runtime that produced these files
# ---------------------------------------------------------------------------


def siesta_runtime_block(fc_run_dir: Path) -> dict[str, Any]:
    """The C01 record, plus the version banner of the run output next to the .FC."""
    runtime = dict(epc_siesta_runtime_block(REPO_ROOT))
    outputs = sorted(path for path in fc_run_dir.glob("*.out") if path.is_file())
    observed = None
    for path in outputs:
        observed = extract_siesta_version_from_text(path.read_text(encoding="utf-8", errors="replace"))
        if observed:
            runtime["run_output"] = {"path": _relative(path), "sha256": file_sha256(path)}
            break
    expected = runtime.get("version_token")
    runtime["run_output_version"] = observed
    if not observed:
        outcome, detail = "absent", f"no SIESTA version banner in {[p.name for p in outputs]}"
    elif not expected:
        outcome, detail = "absent", f"run output says {observed!r}; no C01 record to compare against"
    else:
        outcome = "pass" if observed == expected else "fail"
        detail = f"c01={expected!r} run_output={observed!r}"
    runtime["version_check"] = evidence_item("run_output_version_matches_c01_runtime", outcome, detail)
    if outcome == "fail":
        runtime["status"] = INVALID
    elif runtime.get("status") == PRESENT_VALID and outcome != "pass":
        runtime["status"] = PRESENT_UNVERIFIED
    return runtime


# ---------------------------------------------------------------------------
# Artifact classification
# ---------------------------------------------------------------------------


def previous_declarations(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """What the previous archive of this directory declared, by file name."""
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return {
        str(record["label"]): record
        for record in payload.get("artifacts", [])
        if isinstance(record, dict) and record.get("label") and record.get("declared_sha256")
    }


def archive_artifacts(
    output_dir: Path,
    *,
    copied: dict[str, dict[str, Any]],
    produced: dict[str, str],
    previous: dict[str, dict[str, Any]],
    geometry_signature: str,
    vibra: dict[str, Any],
) -> list[dict[str, Any]]:
    """One classified record per file the archive holds or ever declared."""
    present = {path.name for path in output_dir.iterdir() if path.is_file() and path.name != MANIFEST_NAME}
    records: list[dict[str, Any]] = []
    for name in sorted(present | set(previous)):
        if name in copied:
            entry = copied[name]
            origin = "copied_from_fc_run"
            declared_sha256 = entry["source_sha256"]
            declared_by = entry["source"]
            path_source = f"fc_run_dir_glob:{entry['source']}"
            # The action ("copied"/"unchanged") is deliberately not recorded: it
            # is I/O of this run, and it would make two identical reruns write
            # two different manifests.
            evidence = [
                evidence_item("copied_from_fc_run", "pass", f"hashed at the source {entry['source']}")
            ]
        elif name in produced:
            origin = "produced_by_vibra"
            declared_sha256 = produced[name]
            declared_by = f"{MANIFEST_NAME}#vibra"
            path_source = "vibra_run_in_output_dir"
            evidence = [
                evidence_item(
                    "produced_by_vibra_in_this_run",
                    "pass",
                    f"vibra_bin={vibra.get('vibra_path')} sha256={vibra.get('vibra_sha256')} "
                    f"returncode={vibra.get('vibra_returncode')}; hashed when it appeared and "
                    "rehashed for this manifest",
                )
            ]
        elif name in previous:
            origin = "retained_from_previous_archive"
            declared_sha256 = previous[name]["declared_sha256"]
            declared_by = f"{MANIFEST_NAME}#previous"
            path_source = str(previous[name].get("path_source") or "previous_archive")
            evidence = [
                evidence_item(
                    "retained_from_previous_archive",
                    "pass",
                    f"the FC run no longer offers {name}; the archived copy is kept and rehashed",
                )
            ]
        else:
            origin = "undeclared"
            declared_sha256 = None
            declared_by = None
            path_source = "found_in_output_dir"
            evidence = [
                evidence_item(
                    "declared_by_a_producer",
                    "absent",
                    f"{name} is in the archive but neither the FC run, VIBRA nor a previous "
                    "manifest declares it",
                )
            ]

        record = classify_epc_artifact(
            artifact_kind(name, origin),
            output_dir / name,
            path_source=path_source,
            declared_sha256=declared_sha256,
            declared_by=declared_by,
            evidence=evidence,
            label=name,
            repo_root=REPO_ROOT,
        )
        record["origin"] = origin
        record["geometry_basis_sha256"] = geometry_signature
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preserve .FC/dHSdR.nc with provenance and run VIBRA.")
    parser.add_argument("--fc-run-dir", type=Path, default=REPO_ROOT / "AtomDisplacement" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "comparison" / "phonons")
    parser.add_argument("--run-fdf", type=Path, default=None, help="FC RUN.fdf (default: autodetect)")
    parser.add_argument(
        "--basis-dir",
        type=Path,
        default=None,
        help="directory holding the .ion.xml/.psf of the run (default: the FC run dir)",
    )
    parser.add_argument("--vibra-bin", default="vibra")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_vibra(output_dir: Path, vibra_bin: str, vibra_path: str | None) -> dict[str, Any]:
    """Run VIBRA in the archive, or record why it did not run."""
    record: dict[str, Any] = {
        "vibra_bin": vibra_bin,
        "vibra_available": vibra_path is not None,
        "vibra_path": vibra_path,
        "vibra_sha256": file_sha256(Path(vibra_path)) if vibra_path else None,
        "vibra_returncode": None,
    }
    if vibra_path is None:
        record["warning"] = "VIBRA no esta en PATH; se conservaron los outputs FC sin postproceso."
        return record
    log_path = output_dir / "vibra.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [vibra_path],
            cwd=output_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["vibra_returncode"] = process.returncode
    record["vibra_log"] = str(log_path)
    return record


def archive(
    fc_run_dir: Path,
    output_dir: Path,
    *,
    run_fdf: Path | None = None,
    basis_dir: Path | None = None,
    vibra_bin: str = "vibra",
) -> dict[str, Any]:
    """Archive one FC run idempotently and return the manifest payload."""
    fdf_path = resolve_run_fdf(fc_run_dir, run_fdf)
    geometry = geometry_basis_block(fdf_path, basis_dir or fc_run_dir)
    manifest_path = output_dir / MANIFEST_NAME

    previous_payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    previous_signature = (previous_payload.get("geometry_basis") or {}).get("signature_sha256")
    if previous_signature and previous_signature != geometry["signature_sha256"]:
        raise RuntimeError(
            f"{manifest_path} archives geometry/basis {previous_signature}, the FC run declares "
            f"{geometry['signature_sha256']}. Overwriting would link artifacts of two structures "
            "to one manifest and destroy the previous ones; archive this run in another --output-dir."
        )

    copied = copy_outputs(fc_run_dir, output_dir)

    # VIBRA is what it writes: hashing the archive around the call attributes
    # every new *or rewritten* file to it, so a rerun of a VIBRA that stamps its
    # own output is not mistaken for a corrupted archive. Only paid when VIBRA
    # is actually there.
    vibra_path = shutil.which(vibra_bin)
    before = (
        {path.name: file_sha256(path) for path in output_dir.iterdir() if path.is_file()}
        if vibra_path
        else {}
    )
    vibra = run_vibra(output_dir, vibra_bin, vibra_path)
    produced: dict[str, str] = {}
    if vibra_path:
        for path in sorted(output_dir.iterdir()):
            if not path.is_file() or path.name == MANIFEST_NAME:
                continue
            sha256 = file_sha256(path)
            if before.get(path.name) != sha256:
                produced[path.name] = sha256

    artifacts = archive_artifacts(
        output_dir,
        copied=copied,
        produced=produced,
        previous=previous_declarations(manifest_path),
        geometry_signature=geometry["signature_sha256"],
        vibra=vibra,
    )
    # A record can name a file the archive lost; only an existing one is decodable.
    dhsdr_files = [
        output_dir / record["label"]
        for record in artifacts
        if record["kind"] == "dhsdr" and (output_dir / record["label"]).is_file()
    ]
    fc = fc_parameters(fdf_path)

    summary: dict[str, int] = {}
    for record in artifacts:
        summary[record["status"]] = summary.get(record["status"], 0) + 1

    payload: dict[str, Any] = {
        "schema": ARCHIVE_SCHEMA,
        "fc_run_dir": str(fc_run_dir),
        "output_dir": str(output_dir),
        "fc_files": [str(path) for path in sorted(fc_run_dir.glob("*.FC"))],
        "copied_outputs": [entry["path"] for entry in sorted(copied.values(), key=lambda item: item["path"])],
        "geometry_basis": geometry,
        "geometry_basis_sha256": geometry["signature_sha256"],
        "fc_parameters": fc,
        "dhsdr": dhsdr_block(dhsdr_files[0] if dhsdr_files else None, fc),
        "siesta_runtime": siesta_runtime_block(fc_run_dir),
        "artifacts": artifacts,
        "summary": summary,
        "limitations": list(ARCHIVE_LIMITATIONS),
        **vibra,
    }
    # A MISSING artifact is a lost file; a MISSING C01 record only means the
    # runtime is unidentified, and a dHSdR.nc nobody asked for is not lost.
    contributing = [record["status"] for record in artifacts]
    if payload["dhsdr"].get("expected", True):
        contributing.append(payload["dhsdr"]["status"])
    runtime_status = payload["siesta_runtime"]["status"]
    contributing.append(PRESENT_UNVERIFIED if runtime_status == MISSING else runtime_status)
    payload["status"] = worst_status(contributing)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    if not args.fc_run_dir.exists():
        raise RuntimeError(f"No existe el directorio FC: {args.fc_run_dir}")
    fc_files = sorted(args.fc_run_dir.glob("*.FC"))
    if not fc_files:
        raise RuntimeError(f"No se encontro ningun .FC en {args.fc_run_dir}")

    if args.dry_run:
        fdf_path = resolve_run_fdf(args.fc_run_dir, args.run_fdf)
        print(json.dumps({
            "fc_run_dir": str(args.fc_run_dir),
            "output_dir": str(args.output_dir),
            "fc_files": [str(path) for path in fc_files],
            "vibra_bin": args.vibra_bin,
            "vibra_available": shutil.which(args.vibra_bin) is not None,
            "geometry_basis": geometry_basis_block(fdf_path, args.basis_dir or args.fc_run_dir),
            "fc_parameters": fc_parameters(fdf_path),
        }, indent=2, sort_keys=True))
        return 0

    payload = archive(
        args.fc_run_dir,
        args.output_dir,
        run_fdf=args.run_fdf,
        basis_dir=args.basis_dir,
        vibra_bin=args.vibra_bin,
    )
    print(json.dumps({
        "manifest": payload["manifest_path"],
        "status": payload["status"],
        "summary": payload["summary"],
        "geometry_basis_sha256": payload["geometry_basis_sha256"],
        "dhsdr": payload["dhsdr"]["status"],
        "siesta_runtime": payload["siesta_runtime"]["status"],
        "vibra_returncode": payload["vibra_returncode"],
        "unverified": [record["label"] for record in payload["artifacts"] if record["status"] == PRESENT_UNVERIFIED],
        "invalid": [record["label"] for record in payload["artifacts"] if record["status"] == INVALID],
        "missing": [record["label"] for record in payload["artifacts"] if record["status"] == MISSING],
    }, indent=2))
    print(f"[OK] Manifest fonones escrito en {payload['manifest_path']}")
    if payload["status"] in (INVALID, MISSING):
        return 2
    return 0 if payload["vibra_returncode"] in (None, 0) else int(payload["vibra_returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
