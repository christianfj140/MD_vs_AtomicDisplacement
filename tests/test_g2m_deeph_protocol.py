import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_protocol import SCHEMA_NAME, load_protocol, validate_protocol  # noqa: E402


def valid_protocol() -> dict:
    return {
        "protocol_id": "paper_protocol_unit",
        "version": "1.0",
        "datasets": [
            {
                "dataset_id": "joint_a",
                "dataset_root": "Comparison/datasets/joint_a",
                "benchmark_dataset_manifest": "Comparison/datasets/joint_a/benchmark_dataset_manifest.json",
                "frozen_split_manifest": "Comparison/datasets/joint_a/frozen_split_manifest.json",
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
                    "optim_lr": [0.0003, 0.001],
                    "batch_size": [64, 128, 256],
                    "max_epochs": [200],
                    "hidden_irreps": ["32x0e + 32x1o + 32x2e + 32x3o"],
                    "num_interactions": [3],
                    "correlation": [2],
                    "max_ell": [3],
                },
            },
            "deeph": {
                "enabled": True,
                "search_space": {
                    "learning_rate": [0.0001, 0.0003],
                    "batch_size": [4, 8],
                    "epochs": [200],
                    "atom_fea_len": [64],
                    "edge_fea_len": [128],
                    "num_l": [4],
                    "if_lcmp": [True],
                },
            },
        },
        "selection": {
            "split": "validation",
            "metric": "low_energy_rmse_eV",
            "mode": "min",
            "source": "validation_only",
        },
        "early_stopping": {
            "metric": "low_energy_rmse_eV",
            "mode": "min",
            "patience": 30,
            "min_delta": 0.0,
            "max_epochs": 200,
        },
        "search_policy": {
            "strategy": "latin_hypercube",
            "n_trials_per_model": 20,
            "random_seed": 20260528,
        },
        "budget_policy": {
            "mode": "equal_gpu_hours_per_model",
            "gpu_hours_per_model": 24.0,
        },
        "final_seeds": [0, 1, 2],
        "top_k_selection": {
            "k_per_model": 2,
            "split": "validation",
            "metric": "low_energy_rmse_eV",
            "uses_test_metrics": False,
        },
        "final_test_policy": {
            "policy": "locked_until_final",
            "test_split": "test",
            "locked_during_search": True,
            "evaluate_once_after_selection": True,
        },
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


class Graph2MatDeepHProtocolTests(unittest.TestCase):
    def test_valid_final_benchmark_protocol(self) -> None:
        protocol = validate_protocol(valid_protocol())

        self.assertEqual(protocol["schema"], SCHEMA_NAME)
        self.assertIn("protocol_hash", protocol)
        self.assertEqual(protocol["selection"]["split"], "validation")

    def test_example_protocol_loads(self) -> None:
        protocol = load_protocol(REPO_ROOT / "Comparison" / "config" / "g2m_deeph_paper_protocol_v1_example.json")

        self.assertEqual(protocol["schema"], SCHEMA_NAME)
        self.assertEqual(protocol["budget_policy"]["mode"], "equal_n_trials")
        self.assertLessEqual(
            min(protocol["models"]["graph2mat"]["search_space"]["batch_size"]["choices"]),
            128,
        )

    def test_missing_required_fields_fail(self) -> None:
        protocol = valid_protocol()
        protocol.pop("final_test_policy")

        with self.assertRaisesRegex(RuntimeError, "Missing protocol fields: final_test_policy"):
            validate_protocol(protocol)

    def test_invalid_budget_policy_fails(self) -> None:
        protocol = valid_protocol()
        protocol["budget_policy"] = {"mode": "equal_gpu_hours_per_model"}

        with self.assertRaisesRegex(RuntimeError, "gpu_hours_per_model"):
            validate_protocol(protocol)

        protocol = valid_protocol()
        protocol["budget_policy"] = {"mode": "same_number_of_epochs", "n_trials_per_model": 10}
        with self.assertRaisesRegex(RuntimeError, "budget_policy.mode"):
            validate_protocol(protocol)

    def test_invalid_early_stopping_policy_fails(self) -> None:
        protocol = valid_protocol()
        protocol["early_stopping"]["patience"] = 0

        with self.assertRaisesRegex(RuntimeError, "early_stopping.patience"):
            validate_protocol(protocol)

        protocol = valid_protocol()
        protocol["early_stopping"]["metric"] = "training_loss"
        with self.assertRaisesRegex(RuntimeError, "early_stopping.metric must match selection.metric"):
            validate_protocol(protocol)

    def test_invalid_final_test_policy_fails(self) -> None:
        protocol = valid_protocol()
        protocol["final_test_policy"]["locked_during_search"] = False

        with self.assertRaisesRegex(RuntimeError, "locked_during_search"):
            validate_protocol(protocol)

    def test_model_specific_search_spaces_may_differ(self) -> None:
        protocol = validate_protocol(valid_protocol())
        graph2mat_space = protocol["models"]["graph2mat"]["search_space"]
        deeph_space = protocol["models"]["deeph"]["search_space"]

        self.assertEqual(graph2mat_space["optim_lr"], [0.0003, 0.001])
        self.assertEqual(graph2mat_space["batch_size"], [64, 128, 256])
        self.assertEqual(deeph_space["learning_rate"], [0.0001, 0.0003])
        self.assertNotEqual(graph2mat_space["batch_size"], deeph_space["batch_size"])

    def test_graph2mat_final_protocol_requires_small_batch_search(self) -> None:
        protocol = valid_protocol()
        protocol["models"]["graph2mat"]["search_space"]["batch_size"] = [512, 1024]

        with self.assertRaisesRegex(RuntimeError, "batch size <= 128"):
            validate_protocol(protocol)

    def test_test_metrics_cannot_select_configs(self) -> None:
        protocol = valid_protocol()
        protocol["selection"]["split"] = "test"

        with self.assertRaisesRegex(RuntimeError, "test metrics cannot select configs"):
            validate_protocol(protocol)

        protocol = valid_protocol()
        protocol["top_k_selection"]["uses_test_metrics"] = True
        with self.assertRaisesRegex(RuntimeError, "uses_test_metrics must be false"):
            validate_protocol(protocol)

    def test_deeph_equivalence_policy_is_fail_closed(self) -> None:
        protocol = validate_protocol(valid_protocol())

        self.assertEqual(
            protocol["deeph_comparability"]["adapter_equivalence_policy"],
            "fail_closed_unless_proven",
        )

        invalid = valid_protocol()
        invalid["deeph_comparability"]["adapter_equivalence_policy"] = "warn_only"
        with self.assertRaisesRegex(RuntimeError, "fail_closed_unless_proven"):
            validate_protocol(invalid)

    def test_deeph_search_space_rejects_split_and_physics_keys(self) -> None:
        protocol = valid_protocol()
        protocol["models"]["deeph"]["search_space"]["train_ratio"] = [0.8]

        with self.assertRaisesRegex(RuntimeError, "cannot change split/preprocess/physics"):
            validate_protocol(protocol)

    def test_required_telemetry_fields_must_be_present(self) -> None:
        protocol = valid_protocol()
        protocol["required_telemetry"].remove("gpu_hours")

        with self.assertRaisesRegex(RuntimeError, "required_telemetry is missing: gpu_hours"):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
