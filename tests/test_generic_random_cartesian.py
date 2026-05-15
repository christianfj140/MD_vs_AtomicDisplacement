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


def load_random_cartesian_module():
    scripts_dir = REPO_ROOT / "AtomDisplacement" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "generate_random_cartesian_dataset_generic_test",
        scripts_dir / "generate_random_cartesian_dataset.py",
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
            " 6.0 0.0 0.0",
            " 0.0 6.0 0.0",
            " 0.0 0.0 6.0",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 2.0 0.0 0.0 2",
            " 0.0 2.0 0.0 1",
            " 0.0 0.0 2.0 2",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "MeshCutoff 200 Ry",
            "",
        ]
    )


class GenericRandomCartesianTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.module = load_random_cartesian_module()
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

    def config(self, **random_overrides) -> dict:
        random_config = {
            "recipe": "generic_cartesian_noise",
            "n_structures": 4,
            "max_displacement_ang": 0.04,
            "selected_species": None,
            "min_interatomic_distance_ang": 0.5,
            "remove_center_of_mass_translation": False,
            "seed": 12345,
            "variants_per_family": 1,
            "max_attempts_per_structure": 20,
        }
        random_config.update(random_overrides)
        return {
            "material": {
                "label": "sic",
                "fdf": "materials/sic/RUN.fdf",
                "pseudopotential_dir": "materials/sic/pseudos",
                "basis_dir": "materials/sic/basis",
                "structure_type": "crystal",
            },
            "random_cartesian": random_config,
        }

    def run_generator(self, config: dict, output_name: str = "RandomCartesian_steps") -> dict:
        return self.module.generate_dataset(
            config,
            output_dir=self.root / "dataset" / output_name,
            material_base_dir=self.root,
        )

    def test_non_h2o_generic_random_cartesian_generates_materialized_samples(self) -> None:
        manifest = self.run_generator(self.config())
        dataset_root = self.root / "dataset" / "RandomCartesian_steps"

        self.assertEqual(manifest["recipe"], "generic_cartesian_noise")
        self.assertEqual(manifest["generated_structures"], 4)
        self.assertEqual(manifest["material"]["label"], "sic")
        self.assertTrue((dataset_root / "sample_000001" / "Si.psf").exists())
        self.assertTrue((dataset_root / "sample_000001" / "C.psml").exists())
        self.assertTrue((dataset_root / "basis" / "Si.ion.xml").exists())
        self.assertTrue((dataset_root / "dataset_manifest.json").exists())
        self.assertTrue((dataset_root / "split_manifest_summary.json").exists())
        structure = extract_fdf_structure(dataset_root / "sample_000001" / "RUN.fdf")
        self.assertEqual(structure.atom_count, 4)
        self.assertEqual([species.label for species in structure.species], ["Si", "C"])

    def test_fixed_seed_is_deterministic(self) -> None:
        first = self.run_generator(self.config(), output_name="first")
        second = self.run_generator(self.config(), output_name="second")

        self.assertEqual(first["siesta_input_hashes"], second["siesta_input_hashes"])
        self.assertEqual(
            first["deterministic_hashes"]["sample_family_hashes"],
            second["deterministic_hashes"]["sample_family_hashes"],
        )

    def test_species_filter_only_displaces_selected_species(self) -> None:
        manifest = self.run_generator(
            self.config(selected_species=["C"], n_structures=2),
        )

        self.assertEqual([atom["atom_index"] for atom in manifest["selected_atoms"]], [2, 4])
        dataset_root = self.root / "dataset" / "RandomCartesian_steps"
        for sample in manifest["samples"]:
            metadata = json.loads(
                (dataset_root / sample["sample_id"] / "metadata.json").read_text(encoding="utf-8")
            )
            displacements = metadata["displacements_ang"]
            self.assertEqual(displacements[0], [0.0, 0.0, 0.0])
            self.assertEqual(displacements[2], [0.0, 0.0, 0.0])
            self.assertNotEqual(displacements[1], [0.0, 0.0, 0.0])
            self.assertNotEqual(displacements[3], [0.0, 0.0, 0.0])

    def test_min_distance_guard_fails_when_constraints_are_impossible(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "could not generate a valid structure",
        ):
            self.run_generator(
                self.config(
                    n_structures=1,
                    min_interatomic_distance_ang=10.0,
                    max_attempts_per_structure=2,
                )
            )

    def test_group_metadata_keeps_variants_in_same_split(self) -> None:
        manifest = self.run_generator(
            self.config(n_structures=4, variants_per_family=2),
        )
        summary = manifest["split_summary"]

        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["counts"], {"train": 2, "validation": 2, "test": 0})
        self.assertIn("split_group_id", summary["split_group_keys_used"])
        for group in summary["groups"]:
            self.assertEqual(group["sample_count"], 2)
        group_to_split: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            split_payload = json.loads(
                (
                    self.root
                    / "dataset"
                    / "RandomCartesian_steps"
                    / f"split_manifest_{split_name}.json"
                ).read_text(encoding="utf-8")
            )
            for sample in split_payload["samples"]:
                previous = group_to_split.setdefault(sample["split_group_id"], split_name)
                self.assertEqual(previous, split_name)

    def test_legacy_h2o_components_remain_separate(self) -> None:
        config = self.module.random_cartesian_config(
            {
                "structure": {
                    "random_cartesian": {
                        "recipe": "legacy_components",
                        "n_structures": 1,
                        "components": {
                            "atom_displacement": {"enabled": False},
                            "bond_displacement": {"enabled": True, "bonds": "h2o_oh"},
                            "angle_displacement": {"enabled": False},
                        },
                    }
                }
            }
        )

        self.assertEqual(config["recipe"], "legacy_components")
        self.assertTrue(config["components"]["bond_displacement"]["enabled"])
        self.assertEqual(config["components"]["bond_displacement"]["bonds"], "h2o_oh")


if __name__ == "__main__":
    unittest.main()
