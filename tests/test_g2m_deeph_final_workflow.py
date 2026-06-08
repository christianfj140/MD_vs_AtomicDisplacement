import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_final_workflow import build_evidence_bundle_manifest, parse_args, run_stage  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def protocol_payload() -> dict:
    return {
        "protocol_id": "workflow_protocol_unit",
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
                    "optim_lr": {"choices": [0.0003, 0.001]},
                    "batch_size": {"choices": [64, 128]},
                    "max_epochs": {"value": 20},
                    "hidden_irreps": {"choices": ["16x0e + 16x1o + 16x2e"]},
                    "num_interactions": {"value": 2},
                    "correlation": {"value": 2},
                    "max_ell": {"value": 2},
                },
            },
            "deeph": {
                "enabled": True,
                "search_space": {
                    "learning_rate": {"choices": [0.0001, 0.0003]},
                    "batch_size": {"choices": [2, 4]},
                    "epochs": {"value": 20},
                    "atom_fea_len": {"value": 64},
                    "edge_fea_len": {"value": 128},
                    "num_l": {"value": 4},
                    "if_lcmp": {"value": True},
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
            "patience": 5,
            "min_delta": 0.0,
            "max_epochs": 20,
        },
        "search_policy": {
            "strategy": "random",
            "n_trials_per_model": 2,
            "random_seed": 123,
        },
        "budget_policy": {
            "mode": "equal_n_trials",
            "n_trials_per_model": 2,
        },
        "final_seeds": [0, 1, 2],
        "top_k_selection": {
            "k_per_model": 1,
            "split": "validation",
            "metric": "low_energy_rmse_eV",
            "uses_test_metrics": False,
        },
        "final_evaluation": {
            "primary_metric": "low_energy_rmse_eV",
            "mode": "min",
            "secondary_metrics": [
                "fermi_window_rmse_eV",
                "frontier_window_rmse_eV",
                "dos_wasserstein_eV",
                "h_mae_eV",
            ],
            "practical_match": {
                "relative_gap_max": 1.10,
                "absolute_gap_meV_max": None,
                "requires_cost_noninferior": True,
            },
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


def multi_dataset_protocol_payload() -> dict:
    protocol = protocol_payload()
    protocol["datasets"].append(
        {
            "dataset_id": "joint_b",
            "dataset_root": "/tmp/joint_b",
            "benchmark_dataset_manifest": "/tmp/joint_b/benchmark_dataset_manifest.json",
            "frozen_split_manifest": "/tmp/joint_b/frozen_split_manifest.json",
        }
    )
    return protocol


def search_record(model: str, config_id: str, value: float, *, split: str = "validation") -> dict:
    return {
        "status": "completed",
        "model": model,
        "dataset_id": "joint_a",
        "dataset_root": "/tmp/joint_a",
        "config_id": config_id,
        "config_hash": f"hash-{config_id}",
        "run_root": f"/tmp/runs/{config_id}",
        "metric_split": split,
        "low_energy_rmse_eV": value,
        "common": {"epochs": 20, "seed": 0},
        "overrides": {"hidden_irreps": "16x0e"} if model == "graph2mat" else {"atom_fea_len": 64},
    }


class Graph2MatDeepHFinalWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol_path = self.root / "protocol.json"
        write_json(self.protocol_path, protocol_payload())
        self.workflow = self.root / "workflow"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, *args: str) -> dict:
        return run_stage(parse_args(["--workflow-root", str(self.workflow), *args]))

    def test_validate_protocol_and_generate_search_plan(self) -> None:
        validated = self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        planned = self.run_cli("--stage", "generate-search-plan")

        self.assertEqual(validated["status"], "completed")
        self.assertTrue((self.workflow / "protocol" / "validated_protocol.json").exists())
        self.assertEqual(planned["outputs"]["planned_run_count"], 4)
        self.assertTrue((self.workflow / "search" / "search_plan.json").exists())

    def test_run_search_dry_run_writes_payload_without_launching_runner(self) -> None:
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        self.run_cli("--stage", "generate-search-plan")

        manifest = self.run_cli("--stage", "run-search", "--dry-run")

        payload = self.workflow / "search" / "run_search_payload.json"
        run_manifest = self.workflow / "search" / "run_search_manifest.json"
        self.assertEqual(manifest["status"], "planned_dry_run")
        self.assertTrue(payload.exists())
        payload_data = json.loads(payload.read_text(encoding="utf-8"))
        self.assertEqual(payload_data["selected_dataset_id"], "joint_a")
        self.assertEqual(payload_data["executed_dataset_ids"], ["joint_a"])
        self.assertEqual(json.loads(run_manifest.read_text(encoding="utf-8"))["planned_run_count"], 4)

    def test_multi_dataset_run_search_requires_explicit_dataset_id(self) -> None:
        write_json(self.protocol_path, multi_dataset_protocol_payload())
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        self.run_cli("--stage", "generate-search-plan")

        with self.assertRaisesRegex(RuntimeError, "pass --dataset-id"):
            self.run_cli("--stage", "run-search", "--dry-run")

    def test_multi_dataset_run_search_filters_payload_to_selected_dataset(self) -> None:
        write_json(self.protocol_path, multi_dataset_protocol_payload())
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        planned = self.run_cli("--stage", "generate-search-plan")

        manifest = self.run_cli("--stage", "run-search", "--dry-run", "--dataset-id", "joint_b")

        payload = json.loads((self.workflow / "search" / "run_search_payload.json").read_text(encoding="utf-8"))
        planned_runs = payload["_training_sweep_plan"]["planned_runs"]
        self.assertEqual(planned["outputs"]["planned_run_count"], 8)
        self.assertEqual(manifest["outputs"]["selected_dataset_id"], "joint_b")
        self.assertEqual(manifest["outputs"]["planned_run_count"], 4)
        self.assertEqual(payload["dataset_root"], "/tmp/joint_b")
        self.assertEqual(payload["executed_dataset_ids"], ["joint_b"])
        self.assertEqual({row["dataset_id"] for row in planned_runs}, {"joint_b"})

    def test_select_top_k_requires_search_manifest(self) -> None:
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))

        with self.assertRaisesRegex(RuntimeError, "Missing search training_sweep_manifest"):
            self.run_cli("--stage", "select-top-k")

    def test_select_top_k_rejects_test_metrics_during_search(self) -> None:
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        write_json(
            self.workflow / "search" / "training_sweep_manifest.json",
            {"runs": [search_record("graph2mat", "g_bad", 0.1, split="test")]},
        )

        with self.assertRaisesRegex(RuntimeError, "Test metrics are locked"):
            self.run_cli("--stage", "select-top-k")

    def test_top_k_and_final_seed_plan_are_stage_artifacts(self) -> None:
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        write_json(
            self.workflow / "search" / "training_sweep_manifest.json",
            {
                "runs": [
                    search_record("graph2mat", "g_slow", 0.4),
                    search_record("graph2mat", "g_best", 0.2),
                    search_record("deeph", "d_slow", 0.5),
                    search_record("deeph", "d_best", 0.3),
                ]
            },
        )

        selected = self.run_cli("--stage", "select-top-k")
        robust = self.run_cli("--stage", "generate-final-seeds")

        self.assertEqual(selected["outputs"]["selected_count"], 2)
        self.assertEqual(robust["outputs"]["planned_run_count"], 6)
        self.assertTrue((self.workflow / "selection" / "robust_rerun_plan.json").exists())

    def test_multi_dataset_run_final_requires_explicit_dataset_id(self) -> None:
        write_json(self.protocol_path, multi_dataset_protocol_payload())
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        write_json(
            self.workflow / "selection" / "robust_rerun_plan.json",
            {
                "planned_runs": [
                    {"model": "graph2mat", "dataset_id": "joint_a", "config_id": "g_best"},
                    {"model": "graph2mat", "dataset_id": "joint_b", "config_id": "g_best"},
                ],
                "planned_run_count": 2,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "pass --dataset-id"):
            self.run_cli("--stage", "run-final", "--dry-run")

    def test_multi_dataset_run_final_filters_payload_to_selected_dataset(self) -> None:
        write_json(self.protocol_path, multi_dataset_protocol_payload())
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        write_json(
            self.workflow / "selection" / "robust_rerun_plan.json",
            {
                "planned_runs": [
                    {"model": "graph2mat", "dataset_id": "joint_a", "config_id": "g_a", "dataset_root": "/tmp/joint_a"},
                    {"model": "graph2mat", "dataset_id": "joint_b", "config_id": "g_b", "dataset_root": "/tmp/joint_b"},
                ],
                "planned_run_count": 2,
            },
        )

        manifest = self.run_cli("--stage", "run-final", "--dry-run", "--dataset-id", "joint_b")

        payload = json.loads((self.workflow / "final" / "run_final_payload.json").read_text(encoding="utf-8"))
        planned_runs = payload["_training_sweep_plan"]["planned_runs"]
        self.assertEqual(manifest["outputs"]["selected_dataset_id"], "joint_b")
        self.assertEqual(manifest["outputs"]["planned_run_count"], 1)
        self.assertEqual(payload["dataset_root"], "/tmp/joint_b")
        self.assertEqual({row["dataset_id"] for row in planned_runs}, {"joint_b"})

    def test_run_final_budget_comes_from_robust_plan_not_search_budget(self) -> None:
        protocol = protocol_payload()
        protocol["budget_policy"] = {"mode": "equal_n_trials", "n_trials_per_model": 1}
        write_json(self.protocol_path, protocol)
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        planned_runs = []
        for model in ("deeph", "graph2mat"):
            for seed in (0, 1, 2):
                planned_runs.append(
                    {
                        "model": model,
                        "dataset_id": "joint_a",
                        "dataset_root": "/tmp/joint_a",
                        "config_id": f"{model}_seed{seed}",
                        "selected_config_id": model,
                        "seed": seed,
                    }
                )
        write_json(
            self.workflow / "selection" / "robust_rerun_plan.json",
            {"planned_runs": planned_runs, "planned_run_count": len(planned_runs)},
        )

        self.run_cli("--stage", "run-final", "--dry-run")

        payload = json.loads((self.workflow / "final" / "run_final_payload.json").read_text(encoding="utf-8"))
        budget_policy = payload["_training_sweep_plan"]["budget_policy"]
        self.assertEqual(budget_policy["mode"], "equal_n_trials")
        self.assertEqual(budget_policy["n_trials_per_model"], 3)
        self.assertEqual(budget_policy["source"], "robust_rerun_plan")

    def test_run_final_test_dry_run_does_not_launch_training(self) -> None:
        self.test_top_k_and_final_seed_plan_are_stage_artifacts()
        final_root = self.workflow / "fake_final_run"
        write_json(final_root / "sweep" / "training_sweep_manifest.json", {"runs": []})

        manifest = self.run_cli("--stage", "run-final-test", "--dry-run", "--final-run-root", str(final_root))

        self.assertEqual(manifest["status"], "planned_dry_run")
        final_test_manifest = json.loads((self.workflow / "final_test" / "run_final_test_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(final_test_manifest["status"], "planned_dry_run")
        self.assertIn("No training", final_test_manifest["message"])

    def test_run_final_test_materializes_test_rows_and_evaluate_uses_them(self) -> None:
        self.test_top_k_and_final_seed_plan_are_stage_artifacts()
        robust_plan = json.loads((self.workflow / "selection" / "robust_rerun_plan.json").read_text(encoding="utf-8"))
        final_rows = []
        for row in robust_plan["planned_runs"]:
            final_rows.append(
                {
                    **row,
                    "status": "completed",
                    "metric_split": "test",
                    "low_energy_rmse_eV_mean": 0.2 if row["model"] == "graph2mat" else 0.3,
                    "telemetry": {"gpu_hours_total": 1.0, "peak_gpu_memory_mb": 1000.0},
                    "adapter_equivalence_status": "proven_raw_global_hamiltonian_equivalent"
                    if row["model"] == "deeph"
                    else "",
                    "equivalence_status": "proven" if row["model"] == "deeph" else "",
                    "comparability_status": "valid",
                }
            )
        final_root = self.workflow / "fake_final_run"
        write_json(final_root / "sweep" / "training_sweep_manifest.json", {"runs": final_rows})

        materialized = self.run_cli("--stage", "run-final-test", "--final-run-root", str(final_root))
        evaluated = self.run_cli("--stage", "evaluate-final-test")

        self.assertEqual(materialized["status"], "completed")
        self.assertEqual(evaluated["status"], "completed")
        normalized = json.loads((self.workflow / "final_test" / "sweep" / "training_sweep_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(normalized["runs"])
        self.assertEqual({row["protocol_stage"] for row in normalized["runs"]}, {"final_test"})
        self.assertEqual({row["metric_split"] for row in normalized["runs"]}, {"test"})

    def test_run_final_test_rejects_test_metrics_from_search_stage(self) -> None:
        self.test_top_k_and_final_seed_plan_are_stage_artifacts()
        robust_plan = json.loads((self.workflow / "selection" / "robust_rerun_plan.json").read_text(encoding="utf-8"))
        final_root = self.workflow / "fake_final_run"
        write_json(
            final_root / "sweep" / "training_sweep_manifest.json",
            {
                "runs": [
                    {
                        **robust_plan["planned_runs"][0],
                        "status": "completed",
                        "protocol_stage": "search",
                        "metric_split": "test",
                        "low_energy_rmse_eV_mean": 0.2,
                    }
                ]
            },
        )

        with self.assertRaisesRegex(RuntimeError, "Refusing to promote test metrics"):
            self.run_cli("--stage", "run-final-test", "--final-run-root", str(final_root))

    def test_evaluate_final_test_requires_test_metrics(self) -> None:
        self.test_top_k_and_final_seed_plan_are_stage_artifacts()
        final_root = self.workflow / "fake_final_run"
        write_json(final_root / "sweep" / "training_sweep_manifest.json", {"runs": []})

        with self.assertRaisesRegex(RuntimeError, "final_test requires test metrics"):
            self.run_cli("--stage", "evaluate-final-test", "--final-run-root", str(final_root))

    def test_evaluate_final_test_accepts_selected_test_metrics(self) -> None:
        self.test_top_k_and_final_seed_plan_are_stage_artifacts()
        robust_plan = json.loads((self.workflow / "selection" / "robust_rerun_plan.json").read_text(encoding="utf-8"))
        final_rows = []
        for row in robust_plan["planned_runs"]:
            final_rows.append(
                {
                    **row,
                    "status": "completed",
                    "protocol_stage": "final_test",
                    "metric_split": "test",
                    "low_energy_rmse_eV_mean": 0.2 if row["model"] == "graph2mat" else 0.3,
                    "telemetry": {"gpu_hours_total": 1.0, "peak_gpu_memory_mb": 1000.0},
                    "adapter_equivalence_status": "proven_raw_global_hamiltonian_equivalent"
                    if row["model"] == "deeph"
                    else "",
                    "equivalence_status": "proven" if row["model"] == "deeph" else "",
                    "comparability_status": "valid",
                }
            )
        final_root = self.workflow / "fake_final_run"
        write_json(final_root / "sweep" / "training_sweep_manifest.json", {"runs": final_rows})

        manifest = self.run_cli("--stage", "evaluate-final-test", "--final-run-root", str(final_root))

        self.assertEqual(manifest["status"], "completed")
        self.assertTrue((self.workflow / "final_test" / "final_statistics.json").exists())

    def test_evaluate_final_test_uses_final_metric_not_selection_metric(self) -> None:
        protocol = protocol_payload()
        protocol["selection"]["metric"] = "val_loss"
        protocol["early_stopping"]["metric"] = "val_loss"
        protocol["top_k_selection"]["metric"] = "val_loss"
        protocol["final_evaluation"]["primary_metric"] = "low_energy_rmse_eV"
        write_json(self.protocol_path, protocol)
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        write_json(
            self.workflow / "selection" / "robust_rerun_plan.json",
            {
                "planned_runs": [
                    {"model": "graph2mat", "dataset_id": "joint_a", "config_id": "g_best"},
                    {"model": "deeph", "dataset_id": "joint_a", "config_id": "d_best"},
                ]
            },
        )
        final_root = self.workflow / "fake_final_run"
        write_json(
            final_root / "sweep" / "training_sweep_manifest.json",
            {
                "runs": [
                    {
                        "status": "completed",
                        "model": "graph2mat",
                        "dataset_id": "joint_a",
                        "config_id": "g_best",
                        "seed": 0,
                        "protocol_stage": "final_test",
                        "metric_split": "test",
                        "low_energy_rmse_eV_mean": 0.2,
                    },
                    {
                        "status": "completed",
                        "model": "deeph",
                        "dataset_id": "joint_a",
                        "config_id": "d_best",
                        "seed": 0,
                        "protocol_stage": "final_test",
                        "metric_split": "test",
                        "low_energy_rmse_eV_mean": 0.3,
                        "adapter_equivalence_status": "proven_raw_global_hamiltonian_equivalent",
                        "equivalence_status": "proven",
                        "comparability_status": "valid",
                    },
                ]
            },
        )

        self.run_cli("--stage", "evaluate-final-test", "--final-run-root", str(final_root))

        stats = json.loads((self.workflow / "final_test" / "final_statistics.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["metric"], "low_energy_rmse_eV")
        self.assertNotEqual(stats["metric"], protocol["selection"]["metric"])

    def test_generate_report_uses_final_metric_not_selection_metric(self) -> None:
        protocol = protocol_payload()
        protocol["selection"]["metric"] = "val_loss"
        protocol["early_stopping"]["metric"] = "val_loss"
        protocol["top_k_selection"]["metric"] = "val_loss"
        protocol["final_evaluation"]["primary_metric"] = "low_energy_rmse_eV"
        write_json(self.protocol_path, protocol)
        self.run_cli("--stage", "validate-protocol", "--protocol", str(self.protocol_path))
        final_root = self.workflow / "fake_final_run"
        write_json(
            final_root / "sweep" / "training_sweep_manifest.json",
            {
                "runs": [
                    {
                        "status": "completed",
                        "model": "graph2mat",
                        "dataset_id": "joint_a",
                        "config_id": "g_best",
                        "seed": 0,
                        "protocol_stage": "final_test",
                        "metric_split": "test",
                        "low_energy_rmse_eV_mean": 0.2,
                    }
                ]
            },
        )
        write_json(
            self.workflow / "selection" / "robust_rerun_plan.json",
            {"planned_runs": [{"model": "graph2mat", "dataset_id": "joint_a", "config_id": "g_best"}]},
        )

        self.run_cli("--stage", "generate-report", "--final-run-root", str(final_root))

        report = json.loads((self.workflow / "report" / "report_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["metric"], "low_energy_rmse_eV")
        self.assertNotEqual(report["metric"], protocol["selection"]["metric"])

    def test_evidence_bundle_manifest_records_required_dataset_files(self) -> None:
        protocol = protocol_payload()
        dataset_root = self.root / "joint_a"
        dataset_root_b = self.root / "joint_b"
        dataset_root.mkdir(parents=True)
        dataset_root_b.mkdir(parents=True)
        protocol["datasets"][0]["dataset_root"] = str(dataset_root)
        protocol["datasets"][0]["benchmark_dataset_manifest"] = str(dataset_root / "benchmark_dataset_manifest.json")
        protocol["datasets"][0]["frozen_split_manifest"] = str(dataset_root / "frozen_split_manifest.json")
        protocol["datasets"].append(
            {
                "dataset_id": "joint_b",
                "dataset_root": str(dataset_root_b),
                "benchmark_dataset_manifest": str(dataset_root_b / "benchmark_dataset_manifest.json"),
                "frozen_split_manifest": str(dataset_root_b / "frozen_split_manifest.json"),
            }
        )
        write_json(dataset_root / "benchmark_dataset_manifest.json", {"benchmark_ready": True})
        write_json(dataset_root / "frozen_split_manifest.json", {"split_hash": "abc"})
        write_json(dataset_root / "artifact_validation.json", {"valid": True})
        write_json(dataset_root_b / "benchmark_dataset_manifest.json", {"benchmark_ready": True})
        write_json(dataset_root_b / "frozen_split_manifest.json", {"split_hash": "def"})
        write_json(dataset_root_b / "artifact_validation.json", {"valid": True})
        run_root = self.workflow / "runs" / "final"
        write_json(run_root / "sweep" / "training_sweep_manifest.json", {"runs": []})

        bundle = build_evidence_bundle_manifest(
            workflow_root=self.workflow,
            protocol=protocol,
            run_root=run_root,
            report_outputs={},
        )

        self.assertEqual(bundle["status"], "complete")
        self.assertTrue((self.workflow / "evidence" / "evidence_bundle_manifest.json").exists())
        self.assertEqual(bundle["protocol_dataset_ids"], ["joint_a", "joint_b"])
        labels = {entry["label"] for entry in bundle["files"]}
        self.assertIn("joint_a:artifact_validation", labels)
        self.assertIn("joint_b:artifact_validation", labels)


if __name__ == "__main__":
    unittest.main()
