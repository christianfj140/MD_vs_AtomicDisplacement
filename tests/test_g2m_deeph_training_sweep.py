import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_training_sweep import expand_training_sweep  # noqa: E402


DATASETS = [{"dataset_id": "joint_a", "dataset_root": "/tmp/joint_a"}]


class Graph2MatDeepHTrainingSweepTests(unittest.TestCase):
    def test_common_only_expands_for_both_models(self) -> None:
        sweep = expand_training_sweep(
            {
                "enabled": True,
                "common": {"seeds": [1], "epochs": [10], "learning_rate": [0.001], "batch_size": [2]},
                "graph2mat": {"enabled": True},
                "deeph": {"enabled": True},
            },
            datasets=DATASETS,
        )

        self.assertEqual(len(sweep["planned_runs"]), 2)
        by_model = {row["model"]: row for row in sweep["planned_runs"]}
        self.assertEqual(by_model["graph2mat"]["overrides"]["max_epochs"], 10)
        self.assertEqual(by_model["graph2mat"]["overrides"]["optim_lr"], 0.001)
        self.assertEqual(by_model["graph2mat"]["overrides"]["seed_everything"], 1)
        self.assertEqual(by_model["deeph"]["overrides"]["epochs"], 10)
        self.assertEqual(by_model["deeph"]["overrides"]["learning_rate"], 0.001)
        self.assertEqual(by_model["deeph"]["overrides"]["seed"], 1)

    def test_model_specific_fields_do_not_cross_models(self) -> None:
        sweep = expand_training_sweep(
            {
                "enabled": True,
                "common": {"epochs": [10]},
                "graph2mat": {"hidden_irreps_channels": [4], "max_ell": [2], "loss": ["graph2mat.metrics.block_type_mae"]},
                "deeph": {"atom_fea_len": [64], "edge_fea_len": [128], "num_l": [5]},
            },
            datasets=DATASETS,
        )

        graph2mat = next(row for row in sweep["planned_runs"] if row["model"] == "graph2mat")
        deeph = next(row for row in sweep["planned_runs"] if row["model"] == "deeph")
        self.assertEqual(graph2mat["overrides"]["hidden_irreps"], "4x0e + 4x1o + 4x2e")
        self.assertNotIn("atom_fea_len", graph2mat["overrides"])
        self.assertEqual(deeph["overrides"]["atom_fea_len"], 64)
        self.assertNotIn("hidden_irreps", deeph["overrides"])

    def test_max_runs_guard_and_deterministic_ids(self) -> None:
        payload = {
            "enabled": True,
            "max_runs": 10,
            "common": {"epochs": [10, 20]},
            "graph2mat": {"num_interactions": [1, 2]},
            "deeph": {"enabled": False},
        }
        first = expand_training_sweep(payload, datasets=DATASETS)
        second = expand_training_sweep(payload, datasets=DATASETS)

        self.assertEqual(
            [row["config_id"] for row in first["planned_runs"]],
            [row["config_id"] for row in second["planned_runs"]],
        )
        with self.assertRaisesRegex(RuntimeError, "above max_runs"):
            expand_training_sweep({**payload, "max_runs": 1}, datasets=DATASETS)

    def test_unknown_and_forbidden_fields_fail(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unsupported training_sweep.deeph"):
            expand_training_sweep(
                {"enabled": True, "deeph": {"not_a_key": [1]}, "graph2mat": {"enabled": False}},
                datasets=DATASETS,
            )
        with self.assertRaisesRegex(RuntimeError, "cannot change split/preprocess/physics"):
            expand_training_sweep(
                {"enabled": True, "deeph": {"train_ratio": [0.8]}, "graph2mat": {"enabled": False}},
                datasets=DATASETS,
            )


if __name__ == "__main__":
    unittest.main()
