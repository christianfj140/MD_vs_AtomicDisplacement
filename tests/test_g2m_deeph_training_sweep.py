import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_protocol import validate_protocol  # noqa: E402
from g2m_deeph_training_sweep import expand_training_sweep, training_sweep_from_protocol  # noqa: E402


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

    def test_random_search_is_deterministic_and_exact_per_model(self) -> None:
        payload = {
            "enabled": True,
            "max_runs": 20,
            "search_policy": {"strategy": "random", "n_trials_per_model": 3, "random_seed": 7},
            "common": {"epochs": {"value": 50}, "seed": {"choices": [0, 1]}},
            "graph2mat": {
                "enabled": True,
                "optim_lr": {"distribution": "loguniform", "min": 1e-4, "max": 1e-2},
                "batch_size": {"choices": [128, 256]},
                "max_ell": {"type": "int", "min": 1, "max": 3},
            },
            "deeph": {
                "enabled": True,
                "learning_rate": {"distribution": "loguniform", "min": 3e-5, "max": 1e-3},
                "batch_size": {"choices": [4, 8]},
                "num_l": {"type": "int", "min": 3, "max": 5},
            },
        }

        first = expand_training_sweep(payload, datasets=DATASETS)
        second = expand_training_sweep(payload, datasets=DATASETS)

        self.assertEqual(first["search_policy"]["strategy"], "random")
        self.assertEqual(len(first["planned_runs"]), 6)
        self.assertEqual(first["search_plan"]["planned_run_count"], 6)
        self.assertEqual(first["planned_runs"], second["planned_runs"])
        by_model = {model: [row for row in first["planned_runs"] if row["model"] == model] for model in ("graph2mat", "deeph")}
        self.assertEqual(len(by_model["graph2mat"]), 3)
        self.assertEqual(len(by_model["deeph"]), 3)
        for row in by_model["graph2mat"]:
            self.assertGreaterEqual(row["overrides"]["optim_lr"], 1e-4)
            self.assertLessEqual(row["overrides"]["optim_lr"], 1e-2)
            self.assertIn(row["overrides"]["batch_size"], {128, 256})
            self.assertIn(row["overrides"]["max_ell"], {1, 2, 3})
        for row in by_model["deeph"]:
            self.assertGreaterEqual(row["overrides"]["learning_rate"], 3e-5)
            self.assertLessEqual(row["overrides"]["learning_rate"], 1e-3)
            self.assertIn(row["overrides"]["batch_size"], {4, 8})
            self.assertIn(row["overrides"]["num_l"], {3, 4, 5})

    def test_latin_hypercube_search_is_deterministic(self) -> None:
        payload = {
            "enabled": True,
            "max_runs": 8,
            "search_policy": {"strategy": "latin_hypercube", "n_trials_per_model": 4, "random_seed": 11},
            "graph2mat": {
                "enabled": True,
                "optim_lr": {"distribution": "loguniform", "min": 1e-4, "max": 1e-2},
                "batch_size": {"choices": [128, 256]},
            },
            "deeph": {"enabled": False},
        }

        first = expand_training_sweep(payload, datasets=DATASETS)
        second = expand_training_sweep(payload, datasets=DATASETS)

        self.assertEqual(first["planned_runs"], second["planned_runs"])
        self.assertEqual(len(first["planned_runs"]), 4)
        self.assertEqual(first["search_plan"]["strategy"], "latin_hypercube")
        values = sorted(row["overrides"]["optim_lr"] for row in first["planned_runs"])
        self.assertEqual(len(values), 4)
        self.assertGreaterEqual(values[0], 1e-4)
        self.assertLessEqual(values[-1], 1e-2)

    def test_sampled_search_invalid_distribution_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "loguniform requires 0 < min < max"):
            expand_training_sweep(
                {
                    "enabled": True,
                    "search_policy": {"strategy": "random", "n_trials_per_model": 1, "random_seed": 1},
                    "graph2mat": {
                        "enabled": True,
                        "optim_lr": {"distribution": "loguniform", "min": 0.0, "max": 1e-3},
                    },
                    "deeph": {"enabled": False},
                },
                datasets=DATASETS,
            )

        with self.assertRaisesRegex(RuntimeError, "unsupported distribution"):
            expand_training_sweep(
                {
                    "enabled": True,
                    "search_policy": {"strategy": "random", "n_trials_per_model": 1, "random_seed": 1},
                    "graph2mat": {
                        "enabled": True,
                        "optim_lr": {"distribution": "normal", "min": 0.0, "max": 1.0},
                    },
                    "deeph": {"enabled": False},
                },
                datasets=DATASETS,
            )

    def test_tiny_sample_space_reports_duplicate_configs_without_dropping_trials(self) -> None:
        sweep = expand_training_sweep(
            {
                "enabled": True,
                "max_runs": 10,
                "search_policy": {"strategy": "random", "n_trials_per_model": 3, "random_seed": 2},
                "graph2mat": {
                    "enabled": True,
                    "optim_lr": {"value": 0.001},
                    "batch_size": {"value": 128},
                },
                "deeph": {"enabled": False},
            },
            datasets=DATASETS,
        )

        self.assertEqual(len(sweep["planned_runs"]), 3)
        self.assertEqual(sweep["search_plan"]["duplicate_config_count"], 2)
        self.assertTrue(any(row.get("duplicate_config") for row in sweep["planned_runs"]))

    def test_training_sweep_from_protocol_generates_sampled_plan(self) -> None:
        protocol = validate_protocol(
            {
                "protocol_id": "protocol_search_unit",
                "version": "1.0",
                "datasets": [
                    {
                        "dataset_id": "joint_a",
                        "dataset_root": "/tmp/joint_a",
                        "benchmark_dataset_manifest": "/tmp/joint_a/benchmark_dataset_manifest.json",
                        "frozen_split_manifest": "/tmp/joint_a/frozen_split_manifest.json",
                    }
                ],
                "reference_artifacts": {
                    "required": [
                        "RUN.fdf",
                        "SystemLabel.TSHS",
                        "SystemLabel.TSDE",
                        "SystemLabel.HSX",
                        "SystemLabel.STRUCT_OUT",
                        "SystemLabel.XV",
                        "SystemLabel.ORB_INDX",
                        "metadata.json",
                    ],
                    "forbidden": ["ML_prediction.HSX"],
                    "forbid_as_reference": "ML_prediction.HSX",
                },
                "models": {
                    "graph2mat": {
                        "enabled": True,
                        "search_space": {
                            "optim_lr": {"distribution": "loguniform", "min": 1e-4, "max": 1e-3},
                            "batch_size": {"choices": [128, 256]},
                            "max_epochs": {"value": 100},
                        },
                    },
                    "deeph": {
                        "enabled": True,
                        "search_space": {
                            "learning_rate": {"distribution": "loguniform", "min": 3e-5, "max": 1e-3},
                            "batch_size": {"choices": [4, 8]},
                            "epochs": {"value": 100},
                            "num_l": {"choices": [4, 5]},
                        },
                    },
                },
                "selection": {"split": "validation", "metric": "low_energy_rmse_eV", "mode": "min", "source": "validation_only"},
                "early_stopping": {"metric": "low_energy_rmse_eV", "mode": "min", "patience": 5, "min_delta": 0.0, "max_epochs": 100},
                "search_policy": {"strategy": "random", "n_trials_per_model": 2, "random_seed": 123},
                "budget_policy": {"mode": "equal_n_trials", "n_trials_per_model": 2},
                "final_seeds": [0, 1, 2],
                "top_k_selection": {"k_per_model": 1, "split": "validation", "metric": "low_energy_rmse_eV", "uses_test_metrics": False},
                "final_test_policy": {"policy": "locked_until_final", "test_split": "test", "locked_during_search": True, "evaluate_once_after_selection": True},
                "required_telemetry": [
                    "wall_clock_seconds",
                    "gpu_hours",
                    "peak_gpu_memory_mb",
                    "samples_per_second",
                    "matrix_blocks_per_second",
                    "best_validation_epoch",
                ],
                "deeph_comparability": {
                    "adapter_equivalence_policy": "fail_closed_unless_proven",
                    "robust_winner_requires_proven_equivalence": True,
                    "diagnostic_if_unproven": True,
                },
            }
        )

        sweep_payload = training_sweep_from_protocol(protocol)
        sweep = expand_training_sweep(sweep_payload, datasets=protocol["datasets"])

        self.assertEqual(len(sweep["planned_runs"]), 4)
        self.assertEqual(sweep["search_plan"]["protocol_id"], "protocol_search_unit")
        self.assertEqual(sweep["search_plan"]["n_trials_per_model"], 2)
        models = {row["model"] for row in sweep["planned_runs"]}
        self.assertEqual(models, {"graph2mat", "deeph"})


if __name__ == "__main__":
    unittest.main()
