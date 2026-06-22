from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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


def synthetic_base_fdf(*, include_ghost: bool = False) -> str:
    species_count = 2 if include_ghost else 1
    species_block = [
        "%block ChemicalSpeciesLabel",
        " 1 6 C",
    ]
    if include_ghost:
        species_block.append(" 2 -1 Ghost-H")
    species_block.append("%endblock ChemicalSpeciesLabel")
    return "\n".join(
        [
            "SystemName synthetic base",
            "SystemLabel shared_label",
            f"NumberOfSpecies {species_count}",
            "NumberOfAtoms 2",
            *species_block,
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

    def write_pseudo(self, label: str, *, suffix: str = ".psf") -> Path:
        path = self.dataset_root / f"{label}{suffix}"
        path.write_text(f"pseudo for {label}\n", encoding="utf-8")
        return path

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
        self.write_pseudo("C")

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

    def test_dataset_root_pseudopotential_is_staged_before_siesta_run(self) -> None:
        stencil_root, manifest = self.build_stencil()
        self.write_pseudo("C")
        sample_ids = [record["sample_id"] for record in manifest["samples"]]
        captured_dirs: list[Path] = []

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            captured_dirs.append(cwd)
            self.assertTrue((cwd / "C.psf").exists())
            (cwd / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                source_dataset_root=self.dataset_root,
                max_samples=1,
            )

        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(len(captured_dirs), 1)
        self.assertTrue((captured_dirs[0] / "C.psf").exists())
        self.assertEqual(result["source_dataset_root"], str(self.dataset_root))
        self.assertIn(captured_dirs[0].name, sample_ids)

    def test_ghost_pseudopotential_is_staged_when_species_is_declared(self) -> None:
        (self.sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(include_ghost=True), encoding="utf-8")
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")
        self.write_pseudo("Ghost-H")
        staged = {}

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            staged["files"] = sorted(path.name for path in cwd.iterdir() if path.is_file())
            self.assertTrue((cwd / "C.psf").exists())
            self.assertTrue((cwd / "Ghost-H.psf").exists())
            (cwd / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                source_dataset_root=self.dataset_root,
                max_samples=1,
            )

        self.assertEqual(result["samples_failed"], 0)
        self.assertIn("Ghost-H.psf", staged["files"])

    def test_missing_required_pseudo_fails_with_actionable_error(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            source_dataset_root=self.dataset_root,
            diagnostic_only=True,
            max_samples=1,
        )

        self.assertEqual(result["samples_failed"], 1)
        error = result["rows"][0]["error"]
        self.assertIn("Missing required pseudopotential", error)
        self.assertIn("'C'", error)
        self.assertIn(str(self.dataset_root), error)
        self.assertIn("C.psf", error)
        self.assertIn("C.vps", error)
        self.assertIn("C.psml", error)

    def test_max_samples_limits_reference_samples(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            siesta_command=f"{sys.executable} -c \"from pathlib import Path; Path('siesta.TSHS').write_bytes(b'h')\"",
            max_samples=1,
        )

        self.assertEqual(result["samples_total"], 1)
        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(result["max_samples"], 1)
        self.assertIsNone(result["max_jobs"])

    def test_max_jobs_alias_limits_reference_samples_and_is_recorded(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            siesta_command=f"{sys.executable} -c \"from pathlib import Path; Path('siesta.TSHS').write_bytes(b'h')\"",
            max_jobs=1,
            max_jobs_alias_used=True,
        )

        self.assertEqual(result["samples_total"], 1)
        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(result["max_samples"], 1)
        self.assertEqual(result["max_jobs"], 1)
        self.assertTrue(result["max_jobs_alias_used"])

    def test_siesta_command_runs_as_argv_by_default(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["shell"] = kwargs.get("shell")
            self.assertTrue((Path(kwargs["cwd"]) / "C.psf").exists())
            (Path(kwargs["cwd"]) / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                siesta_command=f"{sys.executable} -c \"print('siesta')\"",
                max_samples=1,
            )

        self.assertIsInstance(captured["command"], list)
        self.assertFalse(captured["shell"])
        self.assertEqual(result["samples_failed"], 0)
        self.assertFalse(result["siesta_shell"])

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
