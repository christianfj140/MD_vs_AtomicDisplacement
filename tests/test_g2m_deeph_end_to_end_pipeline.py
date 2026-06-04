from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Comparison" / "scripts" / "g2m_deeph_end_to_end_pipeline.py"
TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_g2m_deeph_final_workflow import protocol_payload, search_record  # noqa: E402
from test_g2m_deeph_final_workflow import write_json as write_test_json  # noqa: E402


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("g2m_deeph_end_to_end_pipeline_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Graph2MatDeepHEndToEndPipelineSkeletonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workflow = self.root / "workflow"
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text("{}\n", encoding="utf-8")
        self.module = load_pipeline_module()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def parse(self, *args: str):
        return self.module.parse_args(
            [
                "--workflow-root",
                str(self.workflow),
                "--protocol",
                str(self.protocol),
                *args,
            ]
        )

    def test_dry_run_writes_stage_manifests_logs_and_pipeline_state(self) -> None:
        payload = self.module.run_pipeline(
            self.parse("--stages", "validate-protocol,generate-search-plan", "--dry-run")
        )

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["selected_stages"], ["validate-protocol", "generate-search-plan"])
        for stage in ("validate-protocol", "generate-search-plan"):
            manifest_path = self.workflow / "stages" / f"{stage}.json"
            stdout_log = self.workflow / "logs" / f"{stage}.stdout.log"
            stderr_log = self.workflow / "logs" / f"{stage}.stderr.log"
            self.assertTrue(manifest_path.exists(), stage)
            self.assertTrue(stdout_log.exists(), stage)
            self.assertTrue(stderr_log.exists(), stage)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "planned_dry_run")
            self.assertIn("g2m_deeph_final_workflow.py", " ".join(manifest["command"]))
            self.assertIn("--stage", manifest["command"])

        state = json.loads((self.workflow / "pipeline_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["selected_stages"], ["validate-protocol", "generate-search-plan"])

    def test_resume_skips_existing_completed_stage(self) -> None:
        stage_dir = self.workflow / "stages"
        stage_dir.mkdir(parents=True)
        completed = {
            "schema": self.module.SCRIPT_SCHEMA,
            "stage": "validate-protocol",
            "status": "completed",
            "command": ["already", "done"],
            "inputs": {},
            "outputs": {},
            "returncode": 0,
        }
        (stage_dir / "validate-protocol.json").write_text(
            json.dumps(completed, indent=2) + "\n",
            encoding="utf-8",
        )

        payload = self.module.run_pipeline(
            self.parse("--stages", "validate-protocol,generate-search-plan", "--dry-run", "--resume")
        )

        self.assertEqual(payload["status"], "completed")
        validate_manifest = json.loads(
            (stage_dir / "validate-protocol.json").read_text(encoding="utf-8")
        )
        self.assertEqual(validate_manifest["status"], "skipped")
        self.assertIn("Skipped by --resume", validate_manifest["message"])
        generate_manifest = json.loads(
            (stage_dir / "generate-search-plan.json").read_text(encoding="utf-8")
        )
        self.assertEqual(generate_manifest["status"], "planned_dry_run")

    def test_failed_command_writes_failed_manifest_and_logs(self) -> None:
        failing_plan = self.module.CommandPlan(
            command=[
                sys.executable,
                "-c",
                "import sys; print('synthetic stdout'); print('synthetic stderr', file=sys.stderr); sys.exit(3)",
            ],
            message="Synthetic failing command.",
        )

        with mock.patch.object(self.module, "build_stage_plan", return_value=failing_plan):
            with self.assertRaisesRegex(RuntimeError, "return code 3"):
                self.module.run_pipeline(self.parse("--stages", "validate-protocol"))

        manifest = json.loads(
            (self.workflow / "stages" / "validate-protocol.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["returncode"], 3)
        self.assertIn("return code 3", " ".join(manifest["blockers"]))
        self.assertIn(
            "synthetic stdout",
            (self.workflow / "logs" / "validate-protocol.stdout.log").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "synthetic stderr",
            (self.workflow / "logs" / "validate-protocol.stderr.log").read_text(encoding="utf-8"),
        )
        state = json.loads((self.workflow / "pipeline_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")

    def test_stage_ordering_respects_canonical_order_and_range(self) -> None:
        args = self.parse(
            "--stages",
            "run-final,validate-protocol,select-top-k,generate-search-plan",
            "--from-stage",
            "generate-search-plan",
            "--stop-after",
            "select-top-k",
            "--dry-run",
        )

        self.assertEqual(
            self.module.resolve_stage_sequence(args),
            ["generate-search-plan", "select-top-k"],
        )

    def test_unknown_stage_fails_before_running(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unknown pipeline stages"):
            self.module.parse_stage_list("validate-protocol,definitely-not-a-stage")

    def test_select_top_k_subprocess_rejects_test_metrics_before_final_test(self) -> None:
        write_test_json(self.protocol, protocol_payload())
        self.module.run_pipeline(self.parse("--stages", "validate-protocol"))
        write_test_json(
            self.workflow / "search" / "training_sweep_manifest.json",
            {"runs": [search_record("graph2mat", "g_bad", 0.1, split="test")]},
        )

        with self.assertRaisesRegex(RuntimeError, "return code"):
            self.module.run_pipeline(self.parse("--stages", "select-top-k"))

        manifest = json.loads((self.workflow / "stages" / "select-top-k.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertNotEqual(manifest["returncode"], 0)
        combined_logs = (
            (self.workflow / "logs" / "select-top-k.stdout.log").read_text(encoding="utf-8")
            + (self.workflow / "logs" / "select-top-k.stderr.log").read_text(encoding="utf-8")
        )
        self.assertIn("Test metrics are locked", combined_logs)

    def test_run_final_test_subprocess_materializes_rows_without_training(self) -> None:
        write_test_json(self.protocol, protocol_payload())
        self.module.run_pipeline(self.parse("--stages", "validate-protocol"))
        robust_plan = {
            "planned_runs": [
                {"model": "graph2mat", "dataset_id": "joint_a", "config_id": "g_final", "seed": 0},
                {"model": "deeph", "dataset_id": "joint_a", "config_id": "d_final", "seed": 0},
            ],
            "planned_run_count": 2,
        }
        write_test_json(self.workflow / "selection" / "robust_rerun_plan.json", robust_plan)
        final_root = self.workflow / "fake_final_run"
        write_test_json(
            final_root / "sweep" / "training_sweep_manifest.json",
            {
                "runs": [
                    {
                        **robust_plan["planned_runs"][0],
                        "status": "completed",
                        "metric_split": "test",
                        "low_energy_rmse_eV_mean": 0.2,
                    },
                    {
                        **robust_plan["planned_runs"][1],
                        "status": "completed",
                        "metric_split": "test",
                        "low_energy_rmse_eV_mean": 0.3,
                        "adapter_equivalence_status": "proven_raw_global_hamiltonian_equivalent",
                        "equivalence_status": "proven",
                    },
                ]
            },
        )

        payload = self.module.run_pipeline(
            self.parse(
                "--stages",
                "run-final-test",
                "--final-run-root",
                str(final_root),
                "--robust-rerun-plan",
                str(self.workflow / "selection" / "robust_rerun_plan.json"),
            )
        )

        self.assertEqual(payload["status"], "completed")
        manifest = json.loads((self.workflow / "stages" / "run-final-test.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertFalse(manifest["heavy"])
        self.assertIn("g2m_deeph_final_workflow.py", " ".join(manifest["command"]))
        self.assertIn("run-final-test", manifest["command"])
        normalized = json.loads(
            (self.workflow / "final_test" / "sweep" / "training_sweep_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(normalized["runs"]), 2)
        self.assertEqual({row["protocol_stage"] for row in normalized["runs"]}, {"final_test"})
        self.assertEqual({row["metric_split"] for row in normalized["runs"]}, {"test"})
        final_test_manifest = json.loads(
            (self.workflow / "final_test" / "run_final_test_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(final_test_manifest["status"], "completed")
        self.assertEqual(final_test_manifest["source_final_run_root"], str(final_root))

    def _write_strict_summary_inputs(
        self,
        *,
        final_stats_allowed: bool = True,
        gate_allowed: bool = True,
        release_status: str = "complete",
        equivalence_status: str | None = "completed",
    ) -> None:
        write_test_json(
            self.workflow / "final_test" / "final_statistics.json",
            {
                "winner_decision": {
                    "robust_claim_allowed": final_stats_allowed,
                    "precision_winner": "deeph" if final_stats_allowed else None,
                    "gates_failed": [] if final_stats_allowed else ["missing_telemetry"],
                }
            },
        )
        write_test_json(
            self.workflow / "gate_status.json",
            {
                "robust_claim_allowed": gate_allowed,
                "claim_status": "robust_allowed" if gate_allowed else "invalid_telemetry",
                "blockers": [] if gate_allowed else ["telemetry_complete"],
            },
        )
        write_test_json(
            self.workflow / "release_manifest.json",
            {
                "status": release_status,
                "missing_required": [] if release_status == "complete" else ["run_root/ML_prediction.HSX"],
                "forbidden_reference_findings": [],
            },
        )
        if equivalence_status is not None:
            write_test_json(
                self.workflow / "equivalence_strict" / "equivalence_strict_summary.json",
                {"status": equivalence_status, "blockers": [] if equivalence_status == "completed" else ["missing evidence"]},
            )

    def test_strict_summary_blocks_missing_equivalence(self) -> None:
        self._write_strict_summary_inputs(equivalence_status=None)

        with self.assertRaisesRegex(RuntimeError, "missing_equivalence"):
            self.module.run_pipeline(self.parse("--stages", "summary"))

        summary = json.loads((self.workflow / "strict_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["robust_claim_allowed"])
        self.assertTrue(any("missing_equivalence_strict_summary" in item for item in summary["blockers"]))

    def test_strict_summary_blocks_missing_telemetry(self) -> None:
        self._write_strict_summary_inputs(final_stats_allowed=False, gate_allowed=False)

        with self.assertRaisesRegex(RuntimeError, "telemetry"):
            self.module.run_pipeline(self.parse("--stages", "summary"))

        summary = json.loads((self.workflow / "strict_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["robust_claim_allowed"])
        self.assertTrue(any("missing_telemetry" in item or "telemetry_complete" in item for item in summary["blockers"]))

    def test_strict_summary_blocks_missing_release_artifacts(self) -> None:
        self._write_strict_summary_inputs(release_status="invalid")

        with self.assertRaisesRegex(RuntimeError, "release"):
            self.module.run_pipeline(self.parse("--stages", "summary"))

        summary = json.loads((self.workflow / "strict_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["robust_claim_allowed"])
        self.assertTrue(any("release_manifest_missing_required" in item for item in summary["blockers"]))

    def test_final_stats_stage_keeps_top_k_configs_separate(self) -> None:
        write_test_json(self.protocol, protocol_payload())
        robust_rows = []
        metric_rows = []
        for model, selected_config_id, value in (
            ("graph2mat", "g_a", 0.10),
            ("graph2mat", "g_b", 0.40),
            ("deeph", "d_a", 0.30),
        ):
            for seed in (0, 1, 2):
                robust_rows.append(
                    {
                        "model": model,
                        "dataset_id": "joint_a",
                        "config_id": selected_config_id,
                        "selected_config_id": selected_config_id,
                        "seed": seed,
                    }
                )
                row = {
                    "status": "completed",
                    "model": model,
                    "dataset_id": "joint_a",
                    "config_id": selected_config_id,
                    "selected_config_id": selected_config_id,
                    "seed": seed,
                    "protocol_stage": "final_test",
                    "metric_split": "test",
                    "low_energy_rmse_eV_mean": value,
                    "telemetry": {"gpu_hours_total": 1.0, "peak_gpu_memory_mb": 1024},
                }
                if model == "deeph":
                    row.update(
                        {
                            "adapter_equivalence_status": "proven_raw_global_hamiltonian_equivalent",
                            "equivalence_status": "proven",
                            "comparability_status": "raw_global_equivalence_proven",
                            "diagnostic_only": False,
                        }
                    )
                metric_rows.append(row)
        robust_plan = {"planned_runs": robust_rows, "planned_run_count": len(robust_rows)}
        write_test_json(self.workflow / "selection" / "robust_rerun_plan.json", robust_plan)
        write_test_json(self.workflow / "final_test" / "sweep" / "training_sweep_manifest.json", {"runs": metric_rows})

        payload = self.module.run_pipeline(
            self.parse(
                "--stages",
                "final-stats",
                "--robust-rerun-plan",
                str(self.workflow / "selection" / "robust_rerun_plan.json"),
                "--final-run-root",
                str(self.workflow / "final_test"),
            )
        )

        self.assertEqual(payload["status"], "completed")
        stats = json.loads((self.workflow / "final_test" / "final_statistics.json").read_text(encoding="utf-8"))
        by_config = {row["selected_config_id"]: row for row in stats["final_seed_summary"]}
        self.assertEqual(set(by_config), {"g_a", "g_b", "d_a"})
        self.assertAlmostEqual(by_config["g_a"]["mean"], 0.10)
        self.assertAlmostEqual(by_config["g_b"]["mean"], 0.40)
        self.assertEqual(stats["winner_decision"]["dataset_decisions"][0]["winner_config_id"], "g_a")


if __name__ == "__main__":
    unittest.main()
