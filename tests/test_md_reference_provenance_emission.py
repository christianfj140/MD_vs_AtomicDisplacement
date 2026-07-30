"""The MD dataset generator must sign each snapshot's SIESTA reference.

Only run_hamiltonian_derivative_siesta_references.py emitted
``positive_siesta_reference_provenance_v3`` records, so MD datasets carried no signature
and every consumer of choose_reference_matrix (metrics, eigenvalues, ranking, release
manifests) rejected their references with
``missing_matching_positive_siesta_provenance``. That is what left the vacancy->w90
campaign with DeepH-only metrics and 7/7 pairs marked failed.

These tests pin the emission and, crucially, that the emitted signature keeps its
adversarial power: a mutated reference must still be rejected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "shared", REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "MD" / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from reference_provenance import build_positive_reference_provenance  # noqa: E402
from reference_selection import choose_reference_matrix  # noqa: E402

# A real MD snapshot: the fdf carries the Verlet block, which is what marks it as a
# trajectory frame and exempts it from the single-point "Job completed" requirement.
RUN_FDF = (
    "SystemLabel siesta\nNumberOfAtoms 1\nNumberOfSpecies 1\n"
    "MD.TypeOfRun Verlet\nMD.Steps 300\n"
    "%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
    "%block LatticeVectors\n8 0 0\n0 8 0\n0 0 8\n%endblock LatticeVectors\n"
    "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
    "%endblock AtomicCoordinatesAndAtomicSpecies\n"
)
# Cumulative trace of a live run, so no "Job completed" marker.
RUN_OUT_MD = "iscf     Eharris\nSCF cycle converged\n"


def _snapshot(directory: Path, matrix: bytes = b"matrix-a") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "RUN.fdf").write_text(RUN_FDF, encoding="utf-8")
    (directory / "RUN.out").write_text(RUN_OUT_MD, encoding="utf-8")
    (directory / "siesta.TSHS").write_bytes(matrix)
    (directory / "siesta.ORB_INDX").write_bytes(b"orb")
    return directory


def _sign(sample: Path) -> None:
    provenance = build_positive_reference_provenance(
        sample,
        sample / "siesta.TSHS",
        frozen_sample_id=f"md_{sample.name}",
        split="train",
        frozen_split_hash="frozen-split-hash",
        basis_hashes={"C.ion.xml": "basis"},
        pseudopotential_hashes={"C": "pseudo"},
        siesta_version="SIESTA 5.4.2-test",
        siesta_command="siesta < RUN.fdf",
    )
    (sample / "siesta_reference_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def test_unsigned_md_snapshot_is_rejected(tmp_path):
    """Reproduces the campaign failure: no signature, no usable reference."""
    sample = _snapshot(tmp_path / "0")
    selection = choose_reference_matrix(sample)
    assert not selection.ok
    assert "provenance" in selection.reason


def test_signed_md_snapshot_is_accepted(tmp_path):
    """An MD frame signed at generation time must pass, despite no 'Job completed'."""
    sample = _snapshot(tmp_path / "0")
    _sign(sample)
    selection = choose_reference_matrix(sample)
    assert selection.ok, selection.reason
    assert selection.path == sample / "siesta.TSHS"


def test_signature_still_detects_a_mutated_reference(tmp_path):
    """The emission must not blunt the adversarial check it exists for."""
    sample = _snapshot(tmp_path / "0")
    _sign(sample)
    (sample / "siesta.TSHS").write_bytes(b"tampered")

    selection = choose_reference_matrix(sample)
    assert not selection.ok
    assert "reference_sha256_mismatch" in selection.reason


def test_generator_exposes_the_emission_step(tmp_path):
    """write_reference_provenance must be wired into the generator, not just importable."""
    import generate_md_dataset  # noqa: PLC0415

    assert callable(generate_md_dataset.write_reference_provenance)
    source = Path(generate_md_dataset.__file__).read_text(encoding="utf-8")
    # Called after write_benchmark_manifests: the signature needs the frozen split hash.
    assert "write_reference_provenance(" in source
    assert source.index("write_benchmark_manifests(") < source.index(
        "signed = write_reference_provenance("
    )
