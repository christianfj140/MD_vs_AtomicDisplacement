from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"


def load_pipeline_ui_module():
    for path in (SCRIPTS_DIR, SHARED_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "pipeline_ui_material_ui_test",
        SCRIPTS_DIR / "pipeline_ui.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_fdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "SystemName sic",
                "NumberOfAtoms 2",
                "%block ChemicalSpeciesLabel",
                "1 14 Si",
                "2 6 C",
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "1.0 1.0 1.0 2",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
                "",
            ]
        ),
        encoding="utf-8",
    )


class MaterialUiApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_pipeline_ui_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_bundle(self) -> dict[str, str]:
        material_root = self.root / "materials" / "sic"
        write_fdf(material_root / "RUN.fdf")
        pseudo_dir = material_root / "pseudos"
        pseudo_dir.mkdir(parents=True)
        (pseudo_dir / "Si.psf").write_text("si pseudo\n", encoding="utf-8")
        (pseudo_dir / "C.psml").write_text("c pseudo\n", encoding="utf-8")
        basis_dir = material_root / "basis"
        basis_dir.mkdir()
        (basis_dir / "Si.ion.xml").write_text("<ion />\n", encoding="utf-8")
        (basis_dir / "C.ion.xml").write_text("<ion />\n", encoding="utf-8")
        return {
            "label": "sic",
            "fdf": "materials/sic/RUN.fdf",
            "pseudopotential_dir": "materials/sic/pseudos",
            "basis_dir": "materials/sic/basis",
            "structure_type": "crystal",
        }

    def test_material_validation_accepts_h2o_preset(self) -> None:
        payload = self.module.material_validation_response({"preset": "h2o"})

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["material"]["label"], "h2o")
        self.assertEqual(payload["material"]["material_source"], "explicit_preset")
        self.assertEqual([item["label"] for item in payload["species"]], ["O", "H"])
        self.assertEqual(payload["atom_count"], 3)

    def test_material_validation_accepts_explicit_bundle_path(self) -> None:
        payload = self.module.material_validation_response(
            self.write_bundle(),
            base_dir=self.root,
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["material"]["label"], "sic")
        self.assertEqual(payload["atom_count"], 2)
        self.assertEqual([item["label"] for item in payload["species"]], ["Si", "C"])
        self.assertEqual(sorted(payload["pseudopotentials"]), ["C", "Si"])
        self.assertEqual(sorted(payload["basis_files"]), ["C.ion.xml", "Si.ion.xml"])

    def test_invalid_material_does_not_fallback_to_h2o(self) -> None:
        payload = self.module.material_validation_response(
            {
                "label": "sic",
                "fdf": "materials/sic/RUN.fdf",
                "pseudopotential_dir": "materials/sic/pseudos",
            },
            base_dir=self.root,
        )

        self.assertFalse(payload["ok"])
        self.assertIn("FDF does not exist", payload["message"])
        self.assertNotIn("h2o", payload["message"].lower())

    def test_empty_explicit_material_is_rejected_not_treated_as_legacy_default(self) -> None:
        with self.assertRaisesRegex(Exception, "Define material.preset"):
            self.module.parse_material_payload({}, required=True)

    def test_experiment_start_rejects_invalid_material_before_thread_start(self) -> None:
        runner = self.module.ExperimentRunner()

        with self.assertRaisesRegex(Exception, "Material preset 'missing' does not exist"):
            runner.start(
                [3],
                [],
                selected_methods=["md"],
                run_mode="dataset_only",
                material={"preset": "missing"},
            )

        self.assertFalse(runner.status()["running"])

    def test_material_selection_is_written_to_temporary_config(self) -> None:
        config: dict[str, object] = {}
        material = {"preset": "h2o"}

        self.module.apply_material_to_config(config, material)

        self.assertEqual(config["material"], material)

    def test_docs_show_real_material_config_keys(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        for text in (
            "material:",
            "preset: h2o",
            "pseudopotential_dir:",
            "basis_dir:",
            "Validate material",
        ):
            self.assertIn(text, readme)

    def test_ui_contains_material_selector_and_validation_endpoints(self) -> None:
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        backend = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")

        for text in (
            'id="material-mode"',
            'id="material-preset"',
            'id="material-fdf"',
            'id="validate-material"',
        ):
            self.assertIn(text, index_html)
        self.assertIn("/api/material/presets", app_js)
        self.assertIn("/api/material/validate", app_js)
        self.assertIn('elif path == "/api/material/validate"', backend)


if __name__ == "__main__":
    unittest.main()
