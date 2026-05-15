from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "AtomDisplacement" / "scripts"


def load_atom_utils():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "atom_displacement_utils_material_provenance_test",
        SCRIPTS_DIR / "atom_displacement_utils.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_md_generator():
    md_scripts = REPO_ROOT / "MD" / "scripts"
    if str(md_scripts) not in sys.path:
        sys.path.insert(0, str(md_scripts))
    spec = importlib.util.spec_from_file_location(
        "generate_md_dataset_material_provenance_test",
        md_scripts / "generate_md_dataset.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_fdf_text() -> str:
    return "\n".join(
        [
            "SystemName synthetic",
            "SystemLabel synthetic",
            "NumberOfSpecies 2",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 14 Si",
            " 2 6 C",
            "%endblock ChemicalSpeciesLabel",
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 5.0 0.0 0.0",
            " 0.0 5.0 0.0",
            " 0.0 0.0 5.0",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 1.0 0.0 0.0 2",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "Save.HS T",
            "XML.Write T",
            "",
        ]
    )


class SiestaMaterialProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.module = load_atom_utils()
        self.write_material()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_material(self) -> None:
        material_root = self.root / "materials" / "sic"
        material_root.mkdir(parents=True)
        (material_root / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
        pseudo_dir = material_root / "pseudos"
        pseudo_dir.mkdir()
        (pseudo_dir / "Si.psf").write_text("si pseudo\n", encoding="utf-8")
        (pseudo_dir / "C.psml").write_text("c pseudo\n", encoding="utf-8")
        basis_dir = material_root / "basis"
        basis_dir.mkdir()
        (basis_dir / "Si.ion.xml").write_text("<ion />\n", encoding="utf-8")

    def config(self) -> dict:
        return {
            "paths": {
                "run_fdf_name": "RUN.fdf",
                "run_out_name": "RUN.out",
            },
            "material": {
                "label": "sic",
                "fdf": "materials/sic/RUN.fdf",
                "pseudopotential_dir": "materials/sic/pseudos",
                "basis_dir": "materials/sic/basis",
                "structure_type": "crystal",
            },
        }

    def write_sample(self, name: str = "sample_001") -> Path:
        sample = self.root / name
        sample.mkdir()
        (sample / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
        return sample

    def make_valid_outputs(self, sample: Path) -> None:
        (sample / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
        (sample / "siesta.TSHS").write_bytes(b"matrix")

    def test_sample_preparation_copies_synthetic_material_pseudos(self) -> None:
        sample = self.write_sample()

        manifest = self.module.prepare_sample_material_inputs(
            sample,
            self.config(),
            base_dir=self.root,
        )

        self.assertEqual(manifest["label"], "sic")
        self.assertEqual(sorted(manifest["pseudopotentials_copied_to_sample"].values()), ["C.psml", "Si.psf"])
        self.assertTrue((sample / "Si.psf").exists())
        self.assertTrue((sample / "C.psml").exists())
        self.assertEqual(len(manifest["fdf_sha256"]), 64)

    def test_missing_pseudo_fails_before_execution(self) -> None:
        sample = self.write_sample()
        (self.root / "materials" / "sic" / "pseudos" / "C.psml").unlink()

        with self.assertRaisesRegex(RuntimeError, "Missing pseudopotential for species 'C'"):
            self.module.prepare_sample_material_inputs(
                sample,
                self.config(),
                base_dir=self.root,
            )

    def test_execution_metadata_records_material_hashes_flags_scf_and_matrix(self) -> None:
        sample = self.write_sample()
        self.module.prepare_sample_material_inputs(sample, self.config(), base_dir=self.root)
        self.make_valid_outputs(sample)

        validation = self.module.validate_sample_dir(sample)
        self.assertTrue(validation["valid"], validation["validation_reason"])
        metadata = self.module.update_sample_execution_metadata(
            sample,
            validation,
            {"status": "completed", "wall_time_seconds": 0.1},
            self.config(),
            base_dir=self.root,
        )

        self.assertEqual(metadata["material"]["label"], "sic")
        self.assertEqual(len(metadata["material"]["fdf_sha256"]), 64)
        self.assertEqual(sorted(metadata["pseudopotential_sha256"]), ["C", "Si"])
        self.assertEqual(sorted(metadata["basis_file_sha256"]), ["Si.ion.xml"])
        self.assertTrue(metadata["siesta_output_flags"]["valid"])
        self.assertTrue(metadata["siesta_execution"]["job_completed"])
        self.assertTrue(metadata["siesta_execution"]["scf_converged"])
        self.assertTrue(metadata["reference_matrix"]["path"].endswith("siesta.TSHS"))
        self.assertEqual(len(metadata["reference_matrix"]["sha256"]), 64)

    def test_acceptance_rejects_missing_output_failed_scf_and_stale_matrix(self) -> None:
        missing_output = self.write_sample("missing_output")
        (missing_output / "siesta.TSHS").write_bytes(b"matrix")
        self.assertFalse(self.module.validate_sample_dir(missing_output)["valid"])
        self.assertIn("missing_output", self.module.validate_sample_dir(missing_output)["validation_reason"])

        failed_scf = self.write_sample("failed_scf")
        (failed_scf / "RUN.out").write_text("Job completed\n", encoding="utf-8")
        (failed_scf / "siesta.TSHS").write_bytes(b"matrix")
        self.assertFalse(self.module.validate_sample_dir(failed_scf)["valid"])
        self.assertIn("scf_not_converged", self.module.validate_sample_dir(failed_scf)["validation_reason"])

        stale = self.write_sample("stale_matrix")
        self.make_valid_outputs(stale)
        os.utime(stale / "siesta.TSHS", (1000, 1000))
        os.utime(stale / "RUN.fdf", (2000, 2000))
        os.utime(stale / "RUN.out", (3000, 3000))
        self.assertFalse(self.module.validate_sample_dir(stale)["valid"])
        self.assertIn("stale_matrix", self.module.validate_sample_dir(stale)["validation_reason"])

    def test_acceptance_rejects_missing_hamiltonian_output_flags(self) -> None:
        sample = self.write_sample("missing_flags")
        text = (sample / "RUN.fdf").read_text(encoding="utf-8")
        text = text.replace("Save.HS T\n", "").replace("XML.Write T\n", "")
        (sample / "RUN.fdf").write_text(text, encoding="utf-8")
        self.make_valid_outputs(sample)

        validation = self.module.validate_sample_dir(sample)

        self.assertFalse(validation["valid"])
        self.assertIn("missing_hamiltonian_output_flag", validation["validation_reason"])
        self.assertIn("missing_xml_write_flag", validation["validation_reason"])

    def test_h2o_preset_still_prepares_pseudopotentials(self) -> None:
        sample = self.write_sample()

        manifest = self.module.prepare_sample_material_inputs(
            sample,
            {"material": {"preset": "h2o"}},
            base_dir=REPO_ROOT,
        )

        self.assertEqual(manifest["label"], "h2o")
        self.assertTrue((sample / "H.psf").exists())
        self.assertTrue((sample / "O.psf").exists())

    def test_md_material_preparation_copies_pseudos_and_writes_provenance(self) -> None:
        md_module = load_md_generator()
        config = {
            "_config_dir": self.root,
            "paths": {
                "dataset_dir": str(self.root / "MD" / "dataset"),
                "training_dir": str(self.root / "MD" / "training"),
                "run_fdf_name": "RUN.fdf",
                "run_out_name": "RUN.out",
                "training_config_name": "config.yaml",
                "venv_activate": str(self.root / ".venv" / "bin" / "activate"),
            },
            "material": self.config()["material"],
        }

        manifest = md_module.prepare_material_inputs(config)
        dataset_dir = self.root / "MD" / "dataset"

        self.assertEqual(manifest["label"], "sic")
        self.assertTrue((dataset_dir / "Si.psf").exists())
        self.assertTrue((dataset_dir / "C.psml").exists())
        recorded = json.loads((dataset_dir / "material_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["label"], "sic")
        self.assertEqual(sorted(recorded["pseudopotential_sha256"]), ["C", "Si"])


if __name__ == "__main__":
    unittest.main()
