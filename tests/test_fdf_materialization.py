from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fdf_materialization import (  # noqa: E402
    DEFAULT_REQUIRED_OUTPUT_FLAGS,
    FdfMaterializationError,
    extract_bundle_structure,
    extract_fdf_structure,
    materialize_sample_fdf,
)
from material_bundle import validate_material_config  # noqa: E402
from material_presets import resolve_material_bundle  # noqa: E402


def simple_fdf_text(
    *,
    coordinate_format: str = "Ang",
    include_species: bool = True,
    output_flags: bool = False,
) -> str:
    lines = [
        "# user comment preserved",
        "SystemName   synthetic material",
        "SystemLabel  synthetic",
        "NumberOfSpecies  2",
        "NumberOfAtoms    2",
    ]
    if include_species:
        lines.extend(
            [
                "%block ChemicalSpeciesLabel",
                " 1  14  Si",
                " 2   6  C",
                "%endblock ChemicalSpeciesLabel",
            ]
        )
    lines.extend(
        [
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 4.0 0.0 0.0",
            " 0.0 4.0 0.0",
            " 0.0 0.0 4.0",
            "%endblock LatticeVectors",
            f"AtomicCoordinatesFormat {coordinate_format}",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1 # Si",
            " 1.0 1.0 1.0 2 # C",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "MeshCutoff 300 Ry",
            "Custom.User.Setting keep-me",
        ]
    )
    if output_flags:
        lines.extend(
            [
                "SaveHS false",
                "Save.HS F",
                "XML.Write F",
            ]
        )
    return "\n".join(lines) + "\n"


class FdfMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_base_fdf(self, text: str | None = None) -> Path:
        path = self.root / "RUN.fdf"
        path.write_text(text or simple_fdf_text(), encoding="utf-8")
        return path

    def test_valid_simple_fdf_extracts_species_atoms_and_lattice(self) -> None:
        path = self.write_base_fdf()

        structure = extract_fdf_structure(path, structure_type="crystal")

        self.assertEqual(structure.atom_count, 2)
        self.assertEqual([item.label for item in structure.species], ["Si", "C"])
        self.assertEqual(structure.atom_species, [1, 2])
        self.assertEqual(len(structure.lattice_vectors_ang), 3)
        self.assertEqual(structure.structure_type, "crystal")

    def test_lattice_vectors_are_scaled_to_angstrom(self) -> None:
        path = self.write_base_fdf(simple_fdf_text().replace("LatticeConstant 1.0 Ang", "LatticeConstant 2.0 Ang"))

        structure = extract_fdf_structure(path)

        self.assertEqual(structure.lattice_vectors_ang[0], (8.0, 0.0, 0.0))

    def test_materialized_sample_updates_coordinates_and_preserves_settings(self) -> None:
        output_flags_text = simple_fdf_text(output_flags=True)
        base = self.write_base_fdf(output_flags_text)
        self.assertIn("Custom.User.Setting keep-me", output_flags_text)
        output = self.root / "sample" / "RUN.fdf"

        result = materialize_sample_fdf(
            base,
            output,
            positions_ang=[[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]],
            system_label="sample_001",
            system_name="Synthetic sample 001",
        )
        text = output.read_text(encoding="utf-8")

        self.assertIn("# user comment preserved", text)
        self.assertIn("Custom.User.Setting keep-me", text)
        self.assertIn("SystemLabel                      sample_001", text)
        self.assertIn("0.200000000000", text)
        self.assertNotIn(" 0.0 0.0 0.0 1 # Si", text)
        self.assertEqual(result.metadata["atom_count"], 2)
        self.assertEqual(len(result.metadata["base_fdf_sha256"]), 64)
        self.assertEqual(len(result.metadata["materialized_fdf_sha256"]), 64)

    def test_single_point_strips_md_and_lua_directives(self) -> None:
        base = self.write_base_fdf(
            simple_fdf_text()
            + "MD.TypeOfRun Verlet\nMD.Steps 20\nMD.InitialTemperature 450 K\nWriteMDHistory T\nLua.Script md_store.lua\n"
        )
        output = self.root / "single_point" / "RUN.fdf"

        result = materialize_sample_fdf(
            base,
            output,
            positions_ang=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            single_point=True,
        )
        text = output.read_text(encoding="utf-8")

        self.assertNotIn("Verlet", text)
        self.assertNotIn("MD.Steps", text)
        self.assertNotIn("MD.InitialTemperature", text)
        self.assertNotIn("WriteMDHistory", text)
        self.assertNotIn("Lua.Script", text)
        self.assertIn("MD.TypeOfRun", text)
        self.assertIn("CG", text)
        self.assertIn("MD.NumCGsteps", text)
        self.assertTrue(result.metadata["single_point"])

    def test_default_materialization_preserves_md_directives(self) -> None:
        base = self.write_base_fdf(simple_fdf_text() + "MD.TypeOfRun Verlet\nMD.Steps 20\nLua.Script md_store.lua\n")
        output = self.root / "with_md" / "RUN.fdf"

        result = materialize_sample_fdf(base, output, positions_ang=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        text = output.read_text(encoding="utf-8")

        self.assertIn("Verlet", text)
        self.assertIn("Lua.Script", text)
        self.assertFalse(result.metadata["single_point"])

    def test_unsupported_coordinate_format_fails_clearly(self) -> None:
        path = self.write_base_fdf(simple_fdf_text(coordinate_format="Fractional"))

        with self.assertRaisesRegex(FdfMaterializationError, "unsupported AtomicCoordinatesFormat"):
            extract_fdf_structure(path)

    def test_missing_species_block_fails_clearly(self) -> None:
        path = self.write_base_fdf(simple_fdf_text(include_species=False))

        with self.assertRaisesRegex(FdfMaterializationError, "ChemicalSpeciesLabel"):
            extract_fdf_structure(path)

    def test_required_output_flags_are_inserted_or_replaced(self) -> None:
        base = self.write_base_fdf(simple_fdf_text(output_flags=True))
        output = self.root / "sample" / "RUN.fdf"

        materialize_sample_fdf(
            base,
            output,
            positions_ang=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )
        text = output.read_text(encoding="utf-8")

        for key, value in DEFAULT_REQUIRED_OUTPUT_FLAGS.items():
            self.assertIn(f"{key:<32} {value}", text)
        self.assertNotIn("SaveHS false", text)
        self.assertNotIn("Save.HS F", text)
        self.assertNotIn("XML.Write F", text)

    def test_h2o_preset_fixture_can_be_extracted_and_materialized(self) -> None:
        resolved = resolve_material_bundle({"material": {"preset": "h2o"}}, base_dir=REPO_ROOT)
        structure = extract_bundle_structure(resolved.validated)
        output = self.root / "h2o_sample" / "RUN.fdf"
        shifted = [
            [position[0] + 0.01, position[1], position[2]]
            for position in structure.positions_ang
        ]

        result = materialize_sample_fdf(
            resolved.validated.bundle.fdf,
            output,
            positions_ang=shifted,
            structure_type=resolved.validated.bundle.structure_type,
        )

        self.assertEqual(structure.atom_count, 3)
        self.assertEqual([item.label for item in structure.species], ["O", "H"])
        self.assertEqual(result.metadata["structure_type"], "molecule")
        text = output.read_text(encoding="utf-8")
        self.assertIn("SaveHS", text)
        self.assertIn("Save.HS", text)

    def test_non_h2o_validated_bundle_can_be_materialized(self) -> None:
        material_root = self.root / "materials" / "sic"
        fdf = material_root / "RUN.fdf"
        fdf.parent.mkdir(parents=True)
        fdf.write_text(simple_fdf_text(), encoding="utf-8")
        pseudo_dir = material_root / "pseudos"
        pseudo_dir.mkdir()
        (pseudo_dir / "Si.psf").write_text("si\n", encoding="utf-8")
        (pseudo_dir / "C.psml").write_text("c\n", encoding="utf-8")
        validated = validate_material_config(
            {
                "material": {
                    "label": "sic",
                    "fdf": "materials/sic/RUN.fdf",
                    "pseudopotential_dir": "materials/sic/pseudos",
                    "structure_type": "crystal",
                }
            },
            base_dir=self.root,
        )
        structure = extract_bundle_structure(validated)
        output = self.root / "sic_sample" / "RUN.fdf"

        result = materialize_sample_fdf(
            validated.bundle.fdf,
            output,
            positions_ang=structure.positions_ang,
            structure_type=validated.bundle.structure_type,
        )

        self.assertEqual([item["label"] for item in result.metadata["species"]], ["Si", "C"])
        self.assertEqual(result.metadata["structure_type"], "crystal")


if __name__ == "__main__":
    unittest.main()
