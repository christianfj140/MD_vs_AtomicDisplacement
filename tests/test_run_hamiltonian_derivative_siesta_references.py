from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for directory in (SCRIPTS_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_hamiltonian_derivative_stencils import build_derivative_stencils  # noqa: E402
from hamiltonian_derivative_stencil import discover_derivative_stencils  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    run_derivative_siesta_references,
)


def synthetic_base_fdf() -> str:
    return "\n".join(
        [
            "SystemName synthetic base",
            "SystemLabel shared_label",
            "NumberOfSpecies 1",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 6 C",
            "%endblock ChemicalSpeciesLabel",
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 8.0 0.0 0.0",
            " 0.0 8.0 0.0",
            " 0.0 0.0 8.0",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 1.0 0.0 0.0 1",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "",
        ]
    )


class DerivativeSiestaReferenceStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "source_dataset"
        self.sample_dir = self.dataset_root / "splits" / "test" / "base_0"
        self.sample_dir.mkdir(parents=True)
        (self.sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(), encoding="utf-8")
        (self.sample_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": "base_0",
                    "material_label": "synthetic",
                    "material_compatibility_hash": "material-hash",
                    "orbital_ordering_hash": "orbital-hash",
                    "basis_hash": "basis-hash",
                    "pseudopotential_hash": "pseudo-hash",
                }
            ),
            encoding="utf-8",
        )
        (self.dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps({"rows": [{"sample_id": "base_0", "split": "test", "sample_dir": str(self.sample_dir)}]}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build_stencil(self) -> tuple[Path, dict]:
        stencil_root = self.root / "stencils"
        manifest = build_derivative_stencils(
            source_dataset_root=self.dataset_root,
            output_stencil_root=stencil_root,
            split="test",
            method="central",
            delta_ang_values=[0.1],
            atom_indices_zero_based=[1],
            axes=["y"],
            include_base=True,
        )
        return stencil_root, manifest

    def test_skip_if_exists_avoids_rerunning_existing_reference(self) -> None:
        stencil_root, manifest = self.build_stencil()
        sample_id = sorted(record["sample_id"] for record in manifest["samples"])[0]
        reference_dir = stencil_root / "siesta_hamiltonians" / sample_id
        reference_dir.mkdir(parents=True)
        (reference_dir / "siesta.TSHS").write_bytes(b"existing")

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            siesta_command=f"{sys.executable} -c \"import sys; sys.exit(7)\"",
            max_jobs=1,
            skip_if_exists=True,
        )

        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(result["rows"][0]["status"], "skipped_existing")
        self.assertEqual((reference_dir / "siesta.TSHS").read_bytes(), b"existing")

    def test_missing_structure_reports_actionable_error(self) -> None:
        stencil_root = self.root / "broken_stencil"
        bad = stencil_root / "structures" / "bad_sample"
        bad.mkdir(parents=True)
        (bad / "metadata.json").write_text(json.dumps({"sample_id": "bad_sample"}), encoding="utf-8")

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            diagnostic_only=True,
        )

        self.assertEqual(result["samples_failed"], 1)
        self.assertEqual(result["rows"][0]["status"], "error")
        self.assertEqual(result["rows"][0]["error"], "missing_structure_run_fdf")

    def test_failed_siesta_run_is_recorded_and_fails_closed(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        with self.assertRaisesRegex(DerivativeSiestaReferenceError, "failed for"):
            run_derivative_siesta_references(
                stencil_root=stencil_root,
                siesta_command=f"{sys.executable} -c \"import sys; sys.exit(3)\"",
                max_jobs=1,
            )

        manifest_path = stencil_root / "siesta_hamiltonians" / "derivative_siesta_reference_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["samples_failed"], 1)
        self.assertEqual(manifest["rows"][0]["returncode"], 3)
        self.assertEqual(manifest["rows"][0]["error"], "siesta_returncode_nonzero")

    def test_staged_reference_artifacts_are_discoverable(self) -> None:
        stencil_root, manifest = self.build_stencil()
        existing_root = self.root / "existing_references"
        for record in manifest["samples"]:
            ref_dir = existing_root / record["sample_id"]
            ref_dir.mkdir(parents=True)
            (ref_dir / "siesta.TSHS").write_bytes(f"reference {record['sample_id']}".encode("utf-8"))

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            existing_reference_root=existing_root,
        )

        self.assertEqual(result["samples_failed"], 0)
        self.assertTrue(all(row["status"] == "staged" for row in result["rows"]))
        discoveries = discover_derivative_stencils(
            stencil_root,
            method="graph2mat",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )
        self.assertEqual(len(discoveries), 1)
        self.assertIsNotNone(discoveries[0].stencil)
        self.assertIsNotNone(discoveries[0].stencil.siesta_plus)
        self.assertIsNotNone(discoveries[0].stencil.siesta_minus)
        self.assertNotEqual(discoveries[0].stencil.siesta_plus.matrix_path.name, "ML_prediction.HSX")


if __name__ == "__main__":
    unittest.main()
