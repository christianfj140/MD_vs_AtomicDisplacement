from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from material_bundle import MaterialBundleError  # noqa: E402
from material_presets import (  # noqa: E402
    load_material_preset,
    resolve_material_bundle,
)


class MaterialPresetTests(unittest.TestCase):
    def test_explicit_h2o_material_preset_validates(self) -> None:
        resolved = resolve_material_bundle(
            {"material": {"preset": "h2o"}},
            base_dir=REPO_ROOT,
        )
        manifest = resolved.to_manifest_dict()

        self.assertEqual(manifest["label"], "h2o")
        self.assertEqual(manifest["preset"], "h2o")
        self.assertEqual(manifest["material_source"], "explicit_preset")
        self.assertEqual([row["label"] for row in manifest["species"]], ["O", "H"])
        self.assertEqual(sorted(manifest["pseudopotentials"]), ["H", "O"])
        self.assertIn("H.ion.xml", manifest["basis_file_sha256"])
        self.assertIn("O.ion.xml", manifest["basis_file_sha256"])

    def test_pipeline_configs_explicitly_select_supported_preset(self) -> None:
        expected_labels = {
            "MD/pipeline_config.yaml": "graphene",
            "AtomDisplacement/pipeline_config.yaml": "h2o",
        }
        for relative, expected_label in expected_labels.items():
            with self.subTest(relative=relative):
                config = yaml.safe_load((REPO_ROOT / relative).read_text(encoding="utf-8"))
                resolved = resolve_material_bundle(config, base_dir=REPO_ROOT)
                self.assertEqual(resolved.to_manifest_dict()["label"], expected_label)
                self.assertEqual(resolved.source, "explicit_preset")

    def test_legacy_config_without_material_uses_h2o_preset_with_warning(self) -> None:
        config = yaml.safe_load((REPO_ROOT / "MD" / "pipeline_config.yaml").read_text(encoding="utf-8"))
        config.pop("material", None)

        resolved = resolve_material_bundle(config, base_dir=REPO_ROOT)
        manifest = resolved.to_manifest_dict()

        self.assertEqual(manifest["label"], "h2o")
        self.assertEqual(manifest["preset"], "h2o")
        self.assertEqual(manifest["material_source"], "legacy_default_preset")
        self.assertIn("backward-compatible material preset", manifest["warning"])

    def test_legacy_default_can_be_disabled_for_migration(self) -> None:
        with self.assertRaisesRegex(MaterialBundleError, "No material section configured"):
            resolve_material_bundle({}, base_dir=REPO_ROOT, allow_legacy_default=False)

    def test_missing_preset_file_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preset_dir = Path(tmp) / "materials"
            preset_dir.mkdir()
            with self.assertRaisesRegex(MaterialBundleError, "Material preset 'missing' does not exist"):
                resolve_material_bundle(
                    {"material": {"preset": "missing"}},
                    base_dir=REPO_ROOT,
                    preset_dir=preset_dir,
                )

    def test_material_label_is_available_as_manifest_ready_metadata(self) -> None:
        preset = load_material_preset("h2o")
        self.assertEqual(preset["material"]["label"], "h2o")
        resolved = resolve_material_bundle({"material": {"preset": "h2o"}}, base_dir=REPO_ROOT)

        manifest = resolved.to_manifest_dict()

        self.assertEqual(manifest["label"], "h2o")
        self.assertIn("fdf_sha256", manifest)
        self.assertIn("pseudopotential_sha256", manifest)

    def test_generic_material_validator_does_not_hardcode_h2o(self) -> None:
        text = (REPO_ROOT / "shared" / "material_bundle.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("h2o", text)
        self.assertNotIn("water", text)


if __name__ == "__main__":
    unittest.main()
