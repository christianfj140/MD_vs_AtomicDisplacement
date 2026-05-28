import unittest

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_recipe_helpers import dataset_sweep_recipes_from_payload, md_dataset_recipes_to_specs  # noqa: E402


class DatasetRecipeHelpersTests(unittest.TestCase):
    def specs(self, recipes, *, max_datasets=None):
        return md_dataset_recipes_to_specs(
            {"md": recipes},
            split_ratios={"train": 0.6, "validation": 0.2, "test": 0.2},
            max_datasets=max_datasets,
        )

    def test_parses_one_md_dataset_recipe(self):
        info = self.specs(
            [
                {
                    "recipe_id": "md_300K_10",
                    "blocks": [{"block_id": "md_300K", "n_snapshots": 10, "temperature_K": 300}],
                }
            ]
        )

        self.assertEqual(len(info["md_dataset_specs"]), 1)
        self.assertEqual(info["md_dataset_specs"][0]["size"], 10)
        self.assertEqual(info["md_dataset_specs"][0]["temperature_blocks"][0]["temperature_K"], 300.0)

    def test_parses_multiple_md_dataset_recipes(self):
        info = self.specs(
            [
                {"recipe_id": "md_300K_10", "blocks": [{"n_snapshots": 10, "temperature_K": 300}]},
                {"recipe_id": "md_500K_12", "blocks": [{"n_snapshots": 12, "temperature_K": 500}]},
            ]
        )

        self.assertEqual([spec["recipe_id"] for spec in info["md_dataset_specs"]], ["md_300K_10", "md_500K_12"])

    def test_parses_multiblock_two_temperature_recipe(self):
        info = self.specs(
            [
                {
                    "recipe_id": "md_multitemp",
                    "blocks": [
                        {"block_id": "md_300K", "n_snapshots": 5, "temperature_K": 300},
                        {"block_id": "md_500K", "n_snapshots": 7, "temperature_K": 500},
                    ],
                }
            ]
        )

        spec = info["md_dataset_specs"][0]
        self.assertEqual(spec["size"], 12)
        self.assertEqual([block["temperature_K"] for block in spec["temperature_blocks"]], [300.0, 500.0])

    def test_duplicate_recipe_id_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicado"):
            self.specs(
                [
                    {"recipe_id": "same", "blocks": [{"n_snapshots": 10, "temperature_K": 300}]},
                    {"recipe_id": "same", "blocks": [{"n_snapshots": 12, "temperature_K": 400}]},
                ]
            )

    def test_max_datasets_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "max_datasets=1"):
            self.specs(
                [
                    {"recipe_id": "a", "blocks": [{"n_snapshots": 10, "temperature_K": 300}]},
                    {"recipe_id": "b", "blocks": [{"n_snapshots": 12, "temperature_K": 400}]},
                ],
                max_datasets=1,
            )

    def test_invalid_split_size_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "al menos 3"):
            self.specs([{"recipe_id": "tiny", "blocks": [{"n_snapshots": 2, "temperature_K": 300}]}])

    def test_forbidden_siesta_physics_field_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "MeshCutoff"):
            self.specs(
                [
                    {
                        "recipe_id": "bad",
                        "blocks": [{"n_snapshots": 10, "temperature_K": 300, "MeshCutoff": "100 Ry"}],
                    }
                ]
            )

    def test_existing_experiment_style_dataset_recipes_convert_to_md_specs(self):
        info = md_dataset_recipes_to_specs(
            {
                "md": [
                    {
                        "recipe_id": "experiment_style",
                        "label": "Experiment style",
                        "seed": 11,
                        "blocks": [
                            {
                                "block_id": "md_300",
                                "n_snapshots": 6,
                                "temperature_K": 300,
                                "timestep_fs": 1.0,
                                "ensemble": "nve",
                                "thermostat": "none",
                            }
                        ],
                    }
                ]
            },
            split_ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
        )

        spec = info["md_dataset_specs"][0]
        block = spec["temperature_blocks"][0]
        self.assertEqual(block["timestep_fs"], 1.0)
        self.assertEqual(block["ensemble"], "nve")
        self.assertEqual(block["thermostat"], "none")
        self.assertEqual(spec["recipe_metadata"]["seed"], 11)

    def test_generate_new_payload_becomes_single_md_recipe(self):
        enabled, recipes, max_datasets = dataset_sweep_recipes_from_payload(
            {"dataset_mode": "generate_new", "snapshot_count": 12}
        )

        self.assertTrue(enabled)
        self.assertEqual(max_datasets, 20)
        self.assertEqual(len(recipes), 1)
        self.assertEqual(recipes[0]["blocks"][0]["n_snapshots"], 12)
        self.assertEqual(recipes[0]["blocks"][0]["temperature_K"], 300.0)

    def test_generate_new_payload_ignores_empty_disabled_sweep_object(self):
        enabled, recipes, _ = dataset_sweep_recipes_from_payload(
            {
                "dataset_mode": "generate_new",
                "snapshot_count": 12,
                "dataset_sweep": {"enabled": False, "recipes": []},
            }
        )

        self.assertTrue(enabled)
        self.assertEqual(len(recipes), 1)
        self.assertEqual(recipes[0]["blocks"][0]["n_snapshots"], 12)


if __name__ == "__main__":
    unittest.main()
