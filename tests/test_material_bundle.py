from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from material_bundle import (  # noqa: E402
    MaterialBundleError,
    validate_material_config,
)


def write_fdf(path: Path, *, species_rows: list[str] | None = None, coordinate_rows: list[str] | None = None) -> None:
    species_rows = species_rows or [
        "1 14 Si",
        "2 6 C",
    ]
    coordinate_rows = coordinate_rows or [
        "0.0 0.0 0.0 1",
        "1.0 1.0 1.0 2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "SystemName material fixture",
                "SystemLabel material_fixture",
                "%block ChemicalSpeciesLabel",
                *species_rows,
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                *coordinate_rows,
                "%endblock AtomicCoordinatesAndAtomicSpecies",
                "",
            ]
        ),
        encoding="utf-8",
    )


class MaterialBundleValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_valid_bundle(self) -> dict:
        material_root = self.root / "materials" / "sic"
        write_fdf(material_root / "RUN.fdf")
        pseudo_dir = material_root / "pseudos"
        pseudo_dir.mkdir(parents=True)
        (pseudo_dir / "Si.psf").write_text("pseudo Si\n", encoding="utf-8")
        (pseudo_dir / "C.psml").write_text("pseudo C\n", encoding="utf-8")
        return {
            "material": {
                "label": "sic_test",
                "fdf": "materials/sic/RUN.fdf",
                "pseudopotential_dir": "materials/sic/pseudos",
                "structure_type": "crystal",
            }
        }

    def test_valid_minimal_material_bundle_records_species_pseudos_and_hashes(self) -> None:
        config = self.write_valid_bundle()

        validated = validate_material_config(config, base_dir=self.root)
        manifest = validated.to_manifest_dict()

        self.assertEqual(manifest["label"], "sic_test")
        self.assertEqual([row["label"] for row in manifest["species"]], ["Si", "C"])
        self.assertEqual(sorted(manifest["pseudopotentials"]), ["C", "Si"])
        self.assertEqual(len(manifest["fdf_sha256"]), 64)
        self.assertEqual(len(manifest["pseudopotential_sha256"]["Si"]), 64)
        self.assertEqual(
            validated.to_manifest_dict(),
            validate_material_config(config, base_dir=self.root).to_manifest_dict(),
        )

    def test_missing_fdf_fails_with_clear_message(self) -> None:
        config = self.write_valid_bundle()
        Path(self.root / "materials" / "sic" / "RUN.fdf").unlink()

        with self.assertRaisesRegex(MaterialBundleError, "FDF does not exist"):
            validate_material_config(config, base_dir=self.root)

    def test_missing_pseudopotential_directory_fails(self) -> None:
        config = self.write_valid_bundle()
        pseudo_dir = self.root / "materials" / "sic" / "pseudos"
        for child in pseudo_dir.iterdir():
            child.unlink()
        pseudo_dir.rmdir()

        with self.assertRaisesRegex(MaterialBundleError, "pseudopotential directory does not exist"):
            validate_material_config(config, base_dir=self.root)

    def test_missing_pseudo_for_species_fails(self) -> None:
        config = self.write_valid_bundle()
        (self.root / "materials" / "sic" / "pseudos" / "C.psml").unlink()

        with self.assertRaisesRegex(MaterialBundleError, "Missing pseudopotential for species 'C'"):
            validate_material_config(config, base_dir=self.root)

    def test_duplicate_pseudo_for_species_fails_as_ambiguous(self) -> None:
        config = self.write_valid_bundle()
        (self.root / "materials" / "sic" / "pseudos" / "Si.psml").write_text(
            "pseudo Si duplicate\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(MaterialBundleError, "Ambiguous pseudopotential for species 'Si'"):
            validate_material_config(config, base_dir=self.root)

    def test_optional_basis_directory_is_validated_and_hashed(self) -> None:
        config = self.write_valid_bundle()
        basis_dir = self.root / "materials" / "sic" / "basis"
        basis_dir.mkdir()
        (basis_dir / "Si.ion.xml").write_text("<ion><symbol>Si</symbol></ion>\n", encoding="utf-8")
        config["material"]["basis_dir"] = "materials/sic/basis"

        validated = validate_material_config(config, base_dir=self.root)

        self.assertIn("Si.ion.xml", validated.to_manifest_dict()["basis_file_sha256"])

    def test_missing_optional_basis_directory_fails_when_configured(self) -> None:
        config = self.write_valid_bundle()
        config["material"]["basis_dir"] = "materials/sic/missing_basis"

        with self.assertRaisesRegex(MaterialBundleError, "basis directory does not exist"):
            validate_material_config(config, base_dir=self.root)

    def test_path_traversal_outside_base_dir_fails(self) -> None:
        config = self.write_valid_bundle()
        config["material"]["fdf"] = "../outside/RUN.fdf"

        with self.assertRaisesRegex(MaterialBundleError, "escapes its root"):
            validate_material_config(config, base_dir=self.root)

    def test_absolute_paths_are_allowed_by_default_and_recorded(self) -> None:
        config = self.write_valid_bundle()
        config["material"]["fdf"] = str(self.root / "materials" / "sic" / "RUN.fdf")

        validated = validate_material_config(config, base_dir=self.root)

        self.assertTrue(validated.to_manifest_dict()["absolute_paths_used"])
        with self.assertRaisesRegex(MaterialBundleError, "Absolute material bundle paths"):
            validate_material_config(config, base_dir=self.root, allow_absolute_paths=False)

    def test_coordinates_using_undeclared_species_fail(self) -> None:
        config = self.write_valid_bundle()
        write_fdf(
            self.root / "materials" / "sic" / "RUN.fdf",
            coordinate_rows=["0.0 0.0 0.0 1", "1.0 1.0 1.0 3"],
        )

        with self.assertRaisesRegex(MaterialBundleError, "undeclared species indices"):
            validate_material_config(config, base_dir=self.root)

    def test_unsafe_label_fails(self) -> None:
        config = self.write_valid_bundle()
        config["material"]["label"] = "../sic"

        with self.assertRaisesRegex(MaterialBundleError, "material.label"):
            validate_material_config(config, base_dir=self.root)


if __name__ == "__main__":
    unittest.main()
