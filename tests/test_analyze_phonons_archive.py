"""C07 / E-F_001-S9: the FC archive keeps .FC and dHSdR.nc, with provenance.

Every test builds a whole FC run in ``tmp_path``: a RUN.fdf carrying the real FC
directives, the repository's real ``C.ion.xml`` (so the orbital contract is the
production one), a ``.FC``, a ``RUN.out`` with the SIESTA banner, and a
``dHSdR.nc`` written with the fixture writer of the C06 reader tests.

What is checked is the part that can be lost silently: that the derivative file
is archived at all, that a rerun whose source no longer offers it keeps the copy,
that a file nobody declares never reaches PRESENT_VALID, and that an archive can
never end up linking artifacts of two different geometries.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

pytest.importorskip("netCDF4")

import analyze_phonons as ap  # noqa: E402
from run_inventory import (  # noqa: E402
    INVALID,
    MISSING,
    PRESENT_UNVERIFIED,
    PRESENT_VALID,
    _phonon_archiver_evidence,
)
from test_read_siesta_dhsdr import _blocks, write_dhsdr  # noqa: E402

C_ION_XML = REPO_ROOT / "materials" / "graphene_common" / "basis" / "C.ion.xml"

RUN_FDF = """\
SystemName    graphene_fc
SystemLabel   graphene

NumberOfAtoms          2
NumberOfSpecies        1

%block ChemicalSpeciesLabel
  1   6  C
%endblock ChemicalSpeciesLabel

LatticeConstant      1.46700 Ang
%block LatticeVectors
   1.500000000       -0.8660254038        0.0000000000
   1.500000000        0.8660254038        0.0000000000
   0.000000000        0.0000000000       20.0000000000
%endblock LatticeVectors

AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
  {x:.9f}   0.333333333   0.000000000   1
  0.666666667   0.666666667   0.000000000   1
%endblock AtomicCoordinatesAndAtomicSpecies

MD.TypeOfRun                     FC
FC.Displacement                  {displacement}
FC.First                         1
FC.Last                          2
FC.Save.dHS                      T
FC.dHdR.Tolerance                -1.0 Ry/Bohr
FC.dSdR.Tolerance                -1.0 1/Bohr
"""

RUN_OUT = """\
Authorization required, but no authorization protocol specified

Executable      : siesta
Version         : 5.4.2-11-g4e9a46060
Architecture    : x86_64
Compiler version: GNU-13.3.0
"""

NO_VIBRA = "vibra-not-on-this-path"


def make_fc_run(tmp_path, *, displacement="0.04 Bohr", x=0.333333333):
    """A complete FC run directory, dHSdR.nc included."""
    run_dir = tmp_path / "fc_run"
    run_dir.mkdir(parents=True)
    (run_dir / "RUN.fdf").write_text(RUN_FDF.format(displacement=displacement, x=x), encoding="utf-8")
    (run_dir / "RUN.out").write_text(RUN_OUT, encoding="utf-8")
    (run_dir / "graphene.FC").write_text("Force constants matrix\n 0.0 0.0 0.0\n", encoding="utf-8")
    shutil.copy2(C_ION_XML, run_dir / "C.ion.xml")

    orbital_atom = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    write_dhsdr(
        run_dir / "graphene.dHSdR.nc",
        blocks=_blocks(np.random.default_rng(20260901), no_u=8, orbital_atom=orbital_atom),
        no_u=8,
        dx=0.04,
    )
    return run_dir


def archive(tmp_path, run_dir, **kwargs):
    return ap.archive(run_dir, tmp_path / "archive", vibra_bin=NO_VIBRA, **kwargs)


def record_for(payload, name):
    return next(item for item in payload["artifacts"] if item["label"] == name)


def outcome(checks, name):
    return next(item["outcome"] for item in checks if item["check"] == name)


def test_archives_fc_and_dhsdr_with_hashes_and_physics(tmp_path):
    run_dir = make_fc_run(tmp_path)
    payload = archive(tmp_path, run_dir)

    for name in ("graphene.FC", "graphene.dHSdR.nc"):
        record = record_for(payload, name)
        assert (tmp_path / "archive" / name).is_file()
        assert record["status"] == PRESENT_VALID
        assert record["declared_sha256"] == record["observed_sha256"]
        assert record["origin"] == "copied_from_fc_run"
    assert record_for(payload, "graphene.dHSdR.nc")["kind"] == "dhsdr"

    fc = payload["fc_parameters"]
    assert fc["displacement"]["value"] == 0.04
    assert fc["displacement"]["unit"] == "Bohr"
    assert fc["displacement"]["bohr"] == pytest.approx(0.04)
    assert fc["atom_range"] == {"first_atom": 1, "last_atom": 2, "atoms": [1, 2]}
    assert fc["thresholds"]["dHdR"]["value"] == -1.0
    assert fc["thresholds"]["dSdR"]["value"] == -1.0
    assert fc["md_type_of_run"] == "FC"

    dhsdr = payload["dhsdr"]
    assert dhsdr["status"] == PRESENT_VALID
    assert {item["outcome"] for item in dhsdr["cross_checks"]} == {"pass"}
    assert dhsdr["reader"]["displacement"]["bohr"] == pytest.approx(0.04)
    assert dhsdr["reader"]["displaced_atoms"] == [1, 2]


def test_manifest_links_every_artifact_to_one_geometry_and_basis(tmp_path):
    payload = archive(tmp_path, make_fc_run(tmp_path))
    geometry = payload["geometry_basis"]

    assert geometry["orbital_contract"]["orbital_contract_hash"]
    assert geometry["orbital_contract"]["atom_count"] == 2
    assert geometry["orbital_contract"]["orbital_count"] == 8
    assert "C.ion.xml" in geometry["basis_files"]
    assert {record["geometry_basis_sha256"] for record in payload["artifacts"]} == {
        geometry["signature_sha256"]
    }


def test_siesta_runtime_and_run_output_version_are_recorded(tmp_path):
    payload = archive(tmp_path, make_fc_run(tmp_path))
    runtime = payload["siesta_runtime"]

    assert runtime["run_output_version"] == "5.4.2-11-g4e9a46060"
    assert runtime["run_output"]["path"].endswith("RUN.out")
    # An unidentified C01 runtime downgrades provenance; it never claims a match.
    assert runtime["version_check"]["outcome"] in {"pass", "absent", "fail"}
    if runtime["version_check"]["outcome"] != "pass":
        assert runtime["status"] != PRESENT_VALID


def test_compatible_rerun_is_idempotent_and_keeps_dhsdr(tmp_path):
    run_dir = make_fc_run(tmp_path)
    first = archive(tmp_path, run_dir)
    manifest = tmp_path / "archive" / ap.MANIFEST_NAME
    first_bytes = manifest.read_bytes()

    second = archive(tmp_path, run_dir)
    assert manifest.read_bytes() == first_bytes
    assert second["status"] == first["status"]
    assert record_for(second, "graphene.dHSdR.nc")["origin"] == "copied_from_fc_run"

    # The FC run directory is cleaned; the archive must not lose the derivative.
    (run_dir / "graphene.dHSdR.nc").unlink()
    third = archive(tmp_path, run_dir)
    kept = record_for(third, "graphene.dHSdR.nc")
    assert (tmp_path / "archive" / "graphene.dHSdR.nc").is_file()
    assert kept["origin"] == "retained_from_previous_archive"
    assert kept["status"] == PRESENT_VALID
    assert kept["declared_sha256"] == record_for(first, "graphene.dHSdR.nc")["declared_sha256"]


def test_file_without_provenance_stays_unverified(tmp_path):
    run_dir = make_fc_run(tmp_path)
    archive(tmp_path, run_dir)
    (tmp_path / "archive" / "graphene.vectors").write_text("dropped here by hand\n", encoding="utf-8")

    payload = archive(tmp_path, run_dir)
    stray = record_for(payload, "graphene.vectors")
    assert stray["status"] == PRESENT_UNVERIFIED
    assert stray["origin"] == "undeclared"
    assert outcome(stray["evidence"], "declared_by_a_producer") == "absent"


def test_tampered_retained_artifact_is_invalid(tmp_path):
    run_dir = make_fc_run(tmp_path)
    archive(tmp_path, run_dir)
    (run_dir / "graphene.dHSdR.nc").unlink()
    (tmp_path / "archive" / "graphene.dHSdR.nc").write_bytes(b"not the archived bytes")

    payload = archive(tmp_path, run_dir)
    assert record_for(payload, "graphene.dHSdR.nc")["status"] == INVALID
    assert payload["status"] == INVALID


def test_deleted_artifact_is_reported_missing(tmp_path):
    run_dir = make_fc_run(tmp_path)
    archive(tmp_path, run_dir)
    (run_dir / "graphene.dHSdR.nc").unlink()
    (tmp_path / "archive" / "graphene.dHSdR.nc").unlink()

    payload = archive(tmp_path, run_dir)
    assert record_for(payload, "graphene.dHSdR.nc")["status"] == MISSING
    assert payload["status"] == MISSING


def test_geometry_change_is_refused_instead_of_overwriting(tmp_path):
    run_dir = make_fc_run(tmp_path)
    archive(tmp_path, run_dir)
    archived = tmp_path / "archive" / "graphene.dHSdR.nc"
    before = archived.read_bytes()

    moved = make_fc_run(tmp_path / "moved", x=0.3)
    with pytest.raises(RuntimeError, match="geometry/basis"):
        ap.archive(moved, tmp_path / "archive", vibra_bin=NO_VIBRA)
    assert archived.read_bytes() == before


def test_displacement_mismatch_between_fdf_and_dhsdr_is_invalid(tmp_path):
    run_dir = make_fc_run(tmp_path, displacement="0.05 Bohr")
    payload = archive(tmp_path, run_dir)

    assert payload["dhsdr"]["status"] == INVALID
    assert outcome(payload["dhsdr"]["cross_checks"], "displacement_matches_fdf") == "fail"
    assert payload["status"] == INVALID


def test_vibra_output_is_declared_hashed_and_reused(tmp_path, monkeypatch):
    run_dir = make_fc_run(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "vibra"
    # Writes something different every call: a VIBRA that stamps its own output
    # must not make the rerun look like a corrupted archive.
    fake.write_text(
        "#!/bin/sh\nprintf 'omega %s\\n' \"$(date +%s%N)\" > graphene.vectors\n", encoding="utf-8"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    payload = ap.archive(run_dir, tmp_path / "archive", vibra_bin="vibra")
    produced = record_for(payload, "graphene.vectors")
    assert payload["vibra_returncode"] == 0
    assert produced["origin"] == "produced_by_vibra"
    assert produced["status"] == PRESENT_VALID
    assert produced["kind"] == "vibra_output"
    assert produced["declared_sha256"] == produced["observed_sha256"]

    again = ap.archive(run_dir, tmp_path / "archive", vibra_bin="vibra")
    rerun = record_for(again, "graphene.vectors")
    assert rerun["origin"] == "produced_by_vibra"
    assert rerun["status"] == PRESENT_VALID
    assert rerun["declared_sha256"] != produced["declared_sha256"]


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    run_dir = make_fc_run(tmp_path)
    output_dir = tmp_path / "archive"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_phonons.py",
            "--fc-run-dir", str(run_dir),
            "--output-dir", str(output_dir),
            "--vibra-bin", NO_VIBRA,
            "--dry-run",
        ],
    )
    assert ap.main() == 0
    assert not output_dir.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["fc_parameters"]["displacement"]["bohr"] == pytest.approx(0.04)


def test_c02_inventory_sees_the_dhsdr_pattern():
    """The C02 reuse inventory reads this archiver's copy list; C07 closes it."""
    assert _phonon_archiver_evidence(REPO_ROOT)["outcome"] == "pass"
