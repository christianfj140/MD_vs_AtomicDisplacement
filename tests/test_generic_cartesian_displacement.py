from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fdf_materialization import extract_fdf_structure  # noqa: E402


def load_generic_cartesian_module():
    scripts_dir = REPO_ROOT / "AtomDisplacement" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "generate_generic_cartesian_displacement_dataset_test",
        scripts_dir / "generate_generic_cartesian_displacement_dataset.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_fdf_text() -> str:
    return "\n".join(
        [
            "SystemName synthetic crystal",
            "SystemLabel synthetic",
            "NumberOfSpecies 2",
            "NumberOfAtoms 4",
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
            " 0.0 1.0 0.0 1",
            " 0.0 0.0 1.0 2",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "MeshCutoff 200 Ry",
            "",
        ]
    )


class GenericCartesianDisplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.module = load_generic_cartesian_module()
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

    def config(self, **atomic_overrides) -> dict:
        atomic = {
            "recipe": "generic_cartesian",
            "amplitude_ang": 0.03,
            "selected_species": None,
            "include_base": False,
        }
        atomic.update(atomic_overrides)
        return {
            "generation": {"sample_id_format": "sample_{index:04d}"},
            "material": {
                "label": "sic",
                "fdf": "materials/sic/RUN.fdf",
                "pseudopotential_dir": "materials/sic/pseudos",
                "basis_dir": "materials/sic/basis",
                "structure_type": "crystal",
            },
            "atomic_displacement": atomic,
        }

    def test_non_h2o_structure_generates_expected_6n_cartesian_samples(self) -> None:
        output_dir = self.root / "dataset" / "samples"

        manifest = self.module.generate_dataset(
            self.config(),
            output_dir=output_dir,
            base_dir=self.root,
        )

        self.assertEqual(manifest["generated_structures"], 24)
        self.assertEqual(manifest["recipe"], "generic_cartesian")
        self.assertEqual(manifest["axis_order"], ["x", "y", "z"])
        self.assertEqual(manifest["sign_order"], [1, -1])
        first = manifest["samples"][0]
        self.assertEqual(first["sample_id"], "sample_0000")
        self.assertEqual(first["atom_index"], 1)
        self.assertEqual(first["species"], "Si")
        self.assertEqual(first["axis"], "x")
        self.assertEqual(first["sign"], 1)
        self.assertEqual(first["amplitude_ang"], 0.03)
        self.assertEqual(first["split_group_id"], "generic_cartesian_displacement:sic:atom_0001")
        self.assertTrue((output_dir / "sample_0000" / "metadata.json").exists())
        self.assertTrue((output_dir.parent / "samples_manifest.json").exists())
        self.assertTrue((output_dir / "dataset_manifest.json").exists())
        self.assertTrue((output_dir / "basis" / "Si.ion.xml").exists())

        structure = extract_fdf_structure(output_dir / "sample_0000" / "RUN.fdf")
        self.assertEqual(structure.positions_ang[0], (0.03, 0.0, 0.0))
        self.assertEqual(structure.positions_ang[1], (1.0, 0.0, 0.0))
        self.assertEqual(structure.positions_ang[2], (0.0, 1.0, 0.0))
        self.assertEqual(structure.positions_ang[3], (0.0, 0.0, 1.0))
        run_text = (output_dir / "sample_0000" / "RUN.fdf").read_text(encoding="utf-8")
        self.assertIn("Save.HS", run_text)
        self.assertIn("TS.HS.Save", run_text)

    def test_species_filter_restricts_selected_atoms(self) -> None:
        output_dir = self.root / "dataset" / "samples"

        manifest = self.module.generate_dataset(
            self.config(selected_species=["C"]),
            output_dir=output_dir,
            base_dir=self.root,
        )

        self.assertEqual(manifest["generated_structures"], 12)
        self.assertEqual([atom["atom_index"] for atom in manifest["selected_atoms"]], [2, 4])
        self.assertEqual([atom["atom_index"] for atom in manifest["skipped_atoms"]], [1, 3])
        self.assertTrue(all(sample["species"] == "C" for sample in manifest["samples"]))

    def test_sample_metadata_records_displacement_recipe_fields(self) -> None:
        output_dir = self.root / "dataset" / "samples"

        manifest = self.module.generate_dataset(
            self.config(selected_species="Si", amplitude_ang="0.05 Ang"),
            output_dir=output_dir,
            base_dir=self.root,
        )
        metadata = json.loads(
            (output_dir / manifest["samples"][1]["sample_id"] / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(metadata["generation_method"], "generic_cartesian")
        self.assertEqual(metadata["atom_index"], 1)
        self.assertEqual(metadata["species"], "Si")
        self.assertEqual(metadata["axis"], "x")
        self.assertEqual(metadata["sign"], -1)
        self.assertEqual(metadata["amplitude_ang"], 0.05)
        self.assertEqual(metadata["displacement_ang"], [-0.05, 0.0, 0.0])
        self.assertEqual(metadata["method"], "siesta_fc_cartesian")

    def test_include_base_adds_reference_sample(self) -> None:
        output_dir = self.root / "dataset" / "samples"

        manifest = self.module.generate_dataset(
            self.config(include_base=True, selected_species=["Si"]),
            output_dir=output_dir,
            base_dir=self.root,
        )

        self.assertEqual(manifest["generated_structures"], 13)
        self.assertTrue(manifest["samples"][0]["is_reference"])
        self.assertEqual(
            manifest["samples"][0]["split_group_id"],
            "generic_cartesian_displacement:sic:reference",
        )

    def test_invalid_amplitude_fails(self) -> None:
        with self.assertRaisesRegex(
            self.module.GenericCartesianDisplacementError,
            "amplitude_ang must be positive",
        ):
            self.module.generate_dataset(
                self.config(amplitude_ang=0.0),
                output_dir=self.root / "dataset" / "samples",
                base_dir=self.root,
            )

    def test_empty_species_selection_fails(self) -> None:
        with self.assertRaisesRegex(
            self.module.GenericCartesianDisplacementError,
            "selected_species cannot be empty",
        ):
            self.module.generate_dataset(
                self.config(selected_species=[]),
                output_dir=self.root / "dataset" / "samples",
                base_dir=self.root,
            )

    def test_h2o_specific_recipe_is_not_generalized_here(self) -> None:
        with self.assertRaisesRegex(
            self.module.GenericCartesianDisplacementError,
            "only supports atomic_displacement.recipe='generic_cartesian'",
        ):
            self.module.generate_dataset(
                self.config(recipe="h2o_hoh"),
                output_dir=self.root / "dataset" / "samples",
                base_dir=self.root,
            )

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        output_dir = self.root / "dataset" / "samples"
        output_dir.mkdir(parents=True)
        (output_dir / "old.txt").write_text("old\n", encoding="utf-8")

        with self.assertRaisesRegex(
            self.module.GenericCartesianDisplacementError,
            "already exists and is not empty",
        ):
            self.module.generate_dataset(
                self.config(),
                output_dir=output_dir,
                base_dir=self.root,
            )


if __name__ == "__main__":
    unittest.main()
