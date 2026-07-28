from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "shared", REPO_ROOT / "Comparison" / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from reference_provenance import build_positive_reference_provenance  # noqa: E402
from reference_selection import choose_reference_matrix  # noqa: E402


def write_reference(root: Path, sample_id: str, matrix: bytes) -> Path:
    sample = root / sample_id
    sample.mkdir()
    (sample / "RUN.fdf").write_text(
        "SystemLabel siesta\nNumberOfAtoms 1\nNumberOfSpecies 1\n"
        "%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
        "%block LatticeVectors\n8 0 0\n0 8 0\n0 0 8\n%endblock LatticeVectors\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n",
        encoding="utf-8",
    )
    (sample / "RUN.out").write_text(
        "iscf Eharris\nSCF cycle converged\nJob completed\n",
        encoding="utf-8",
    )
    reference = sample / "siesta.TSHS"
    reference.write_bytes(matrix)
    (sample / "siesta.ORB_INDX").write_bytes(b"orb")
    provenance = build_positive_reference_provenance(
        sample,
        reference,
        frozen_sample_id=sample_id,
        split="test",
        frozen_split_hash="frozen-split",
        basis_hashes={"C.ion.xml": "basis"},
        pseudopotential_hashes={"C": "pseudo"},
        siesta_version="SIESTA 5.4.2-test",
        siesta_command="siesta < RUN.fdf",
    )
    (sample / "siesta_reference_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    return sample


class ReferenceProvenanceAdversarialTests(unittest.TestCase):
    def test_one_byte_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = write_reference(Path(tmp), "sample_a", b"matrix-a")
            (sample / "siesta.TSHS").write_bytes(b"matrix-b")

            selection = choose_reference_matrix(sample)

            self.assertFalse(selection.ok)
            self.assertIn("reference_sha256_mismatch", selection.reason)

    def test_swapped_reference_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_a = write_reference(root, "sample_a", b"matrix-a")
            sample_b = write_reference(root, "sample_b", b"matrix-b")
            a = (sample_a / "siesta.TSHS").read_bytes()
            b = (sample_b / "siesta.TSHS").read_bytes()
            (sample_a / "siesta.TSHS").write_bytes(b)
            (sample_b / "siesta.TSHS").write_bytes(a)

            self.assertFalse(choose_reference_matrix(sample_a).ok)
            self.assertFalse(choose_reference_matrix(sample_b).ok)


if __name__ == "__main__":
    unittest.main()
