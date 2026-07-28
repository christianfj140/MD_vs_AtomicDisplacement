import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
for directory in (SCRIPTS_DIR, REPO_ROOT / "shared"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deeph_prediction_adapter import EQUIVALENCE_PROVEN_RAW_GLOBAL  # noqa: E402
from g2m_deeph_gate_check import build_gate_status, main  # noqa: E402
from reference_provenance import build_positive_reference_provenance  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_snapshot(path: Path, label: str) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text(
        f"SystemLabel {label}\nNumberOfAtoms 1\nNumberOfSpecies 1\n"
        "%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
        "%block LatticeVectors\n8 0 0\n0 8 0\n0 0 8\n%endblock LatticeVectors\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n",
        encoding="utf-8",
    )
    write_json(path / "metadata.json", {"system_label": label})
    for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
        (path / f"{label}{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
    (path / "RUN.out").write_text(
        "iscf     Eharris\nSCF cycle converged\nJob completed\n",
        encoding="utf-8",
    )
    write_json(
        path / "siesta_reference_provenance.json",
        build_positive_reference_provenance(
            path,
            path / f"{label}.TSHS",
            frozen_sample_id=path.name,
            split="test",
            frozen_split_hash="split-a",
            basis_hashes={"C.ion.xml": "basis-hash"},
            pseudopotential_hashes={"C": "pseudo-hash"},
            siesta_version="SIESTA 5.4.2-test",
            siesta_command="siesta < RUN.fdf",
        ),
    )
    return {
        "snapshot_dir": str(path),
        "valid": True,
        "present_artifacts": {"hsx": str(path / f"{label}.HSX")},
    }


def protocol_payload(dataset_root: Path) -> dict:
    return {
        "protocol_id": "gate_check_protocol_unit",
        "version": "1.0",
        "datasets": [
            {
                "dataset_id": "joint_a",
                "dataset_root": str(dataset_root),
                "benchmark_dataset_manifest": str(dataset_root / "benchmark_dataset_manifest.json"),
                "frozen_split_manifest": str(dataset_root / "frozen_split_manifest.json"),
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
                    "optim_lr": {"choices": [0.001, 0.003]},
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
            "metric": "val_loss",
            "mode": "min",
            "source": "validation_only",
        },
        "early_stopping": {
            "metric": "val_loss",
            "mode": "min",
            "patience": 5,
            "min_delta": 0.0,
            "max_epochs": 20,
        },
        "search_policy": {"strategy": "random", "n_trials_per_model": 2, "random_seed": 1},
        "budget_policy": {"mode": "equal_n_trials", "n_trials_per_model": 2},
        "final_seeds": [0, 1, 2, 3, 4],
        "top_k_selection": {
            "k_per_model": 1,
            "split": "validation",
            "metric": "val_loss",
            "uses_test_metrics": False,
        },
        "final_evaluation": {
            "primary_metric": "low_energy_rmse_eV",
            "mode": "min",
            "secondary_metrics": ["fermi_window_rmse_eV", "dos_wasserstein_eV", "h_mae_eV"],
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


class Graph2MatDeepHGateCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "dataset"
        self.workflow_root = self.root / "workflow"
        self.run_root = self.workflow_root / "runs" / "final"
        self.protocol_path = self.root / "protocol.json"
        self.output_path = self.root / "gate_status.json"
        self.dataset_root.mkdir(parents=True)
        self.run_root.mkdir(parents=True)
        self.write_complete_fixture()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_complete_fixture(self) -> None:
        write_json(self.protocol_path, protocol_payload(self.dataset_root))
        planned = [
            {"model": model, "config_id": f"{model}_frozen_seed{seed}", "seed": seed}
            for model in ("graph2mat", "deeph")
            for seed in range(5)
        ]
        write_json(
            self.workflow_root / "selection" / "selected_configs.json",
            {
                "protocol_stage": "search",
                "uses_test_metrics": False,
                "checkpoint_selection_complete": True,
                "paper_level_blockers": [],
                "selected_configs": [
                    {
                        "model": model,
                        "config_id": f"{model}_frozen",
                        "checkpoint_selection": {"status": "valid"},
                    }
                    for model in ("graph2mat", "deeph")
                ],
            },
        )
        write_json(
            self.workflow_root / "selection" / "robust_rerun_plan.json",
            {
                "protocol_stage": "robust_validation",
                "uses_test_metrics": False,
                "paper_level_blockers": [],
                "planned_runs": planned,
            },
        )
        write_json(
            self.workflow_root / "final_test" / "run_final_test_manifest.json",
            {
                "status": "completed",
                "evaluated_runs": planned,
            },
        )
        snapshots = [
            write_snapshot(self.dataset_root / f"s{index}", f"s{index}")
            for index in range(3)
        ]
        write_json(
            self.dataset_root / "benchmark_dataset_manifest.json",
            {
                "schema": "joint_graph2mat_deeph_benchmark_manifest_v1",
                "benchmark_ready": True,
                "validation_status": "valid",
                "material_profile": "production",
                "frozen_split_manifest": {
                    "path": str(self.dataset_root / "frozen_split_manifest.json"),
                    "split_counts": {"train": 1, "validation": 1, "test": 1},
                    "valid": True,
                },
            },
        )
        write_json(
            self.dataset_root / "frozen_split_manifest.json",
            {
                "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
                "valid": True,
                "split_hash": "split-hash",
                "split_counts": {"train": 1, "validation": 1, "test": 1},
                "rows": [
                    {
                        "sample_id": "s0",
                        "sample_dir": str(self.dataset_root / "s0"),
                        "split": "train",
                        "artifact_paths": {"reference_hsx": str(self.dataset_root / "s0" / "s0.HSX")},
                    },
                    {
                        "sample_id": "s1",
                        "sample_dir": str(self.dataset_root / "s1"),
                        "split": "validation",
                        "artifact_paths": {"reference_hsx": str(self.dataset_root / "s1" / "s1.HSX")},
                    },
                    {
                        "sample_id": "s2",
                        "sample_dir": str(self.dataset_root / "s2"),
                        "split": "test",
                        "artifact_paths": {"reference_hsx": str(self.dataset_root / "s2" / "s2.HSX")},
                    },
                ],
            },
        )
        write_json(
            self.dataset_root / "artifact_validation.json",
            {
                "valid": True,
                "snapshots": snapshots,
            },
        )
        write_json(
            self.workflow_root / "final_test" / "final_statistics.json",
            {
                "schema": "graph2mat_deeph_final_statistics_v1",
                "expected_seeds": [0, 1, 2, 3, 4],
                "final_seed_summary": [
                    {
                        "model": "graph2mat",
                        "dataset_id": "joint_a",
                        "n_seeds_completed": 5,
                        "gpu_hours_mean": 1.0,
                        "peak_gpu_memory_mb_mean": 1000.0,
                    },
                    {
                        "model": "deeph",
                        "dataset_id": "joint_a",
                        "n_seeds_completed": 5,
                        "gpu_hours_mean": 1.2,
                        "peak_gpu_memory_mb_mean": 1200.0,
                        "robust_claim_allowed_by_comparability": True,
                    },
                ],
                "winner_decision": {"robust_claim_allowed": True, "gates_failed": []},
            },
        )
        evidence = self.run_root / "deeph" / "raw_global_equivalence_evidence.json"
        write_json(evidence, {"equivalence_status": "proven", "equivalence_scope": "raw_global"})
        adapter = self.run_root / "deeph" / "adapter_manifest.json"
        write_json(
            adapter,
            {
                "schema": "deeph_hdf5_prediction_adapter_v1",
                "robust_matrix_metrics_allowed": True,
                "adapter_equivalence_statuses": [EQUIVALENCE_PROVEN_RAW_GLOBAL],
                "equivalence_statuses": ["proven"],
                "equivalence_scopes": ["raw_global"],
                "equivalence_evidence_paths": [str(evidence)],
                "equivalence_gate": {"robust_claim_allowed": True, "diagnostic_only": False},
            },
        )
        required_files = [
            ("protocol", self.protocol_path),
            ("benchmark_dataset_manifest", self.dataset_root / "benchmark_dataset_manifest.json"),
            ("frozen_split_manifest", self.dataset_root / "frozen_split_manifest.json"),
            ("artifact_validation", self.dataset_root / "artifact_validation.json"),
            ("final_statistics", self.workflow_root / "final_test" / "final_statistics.json"),
            ("deeph_adapter_manifest", adapter),
        ]
        write_json(
            self.workflow_root / "evidence" / "evidence_bundle_manifest.json",
            {
                "schema": "graph2mat_deeph_final_evidence_bundle_v1",
                "status": "complete",
                "missing_required": [],
                "files": [
                    {
                        "label": label,
                        "path": str(path),
                        "required": True,
                        "exists": path.exists(),
                    }
                    for label, path in required_files
                ],
            },
        )

    def gate_status(self) -> dict:
        return build_gate_status(
            protocol_path=self.protocol_path,
            workflow_root=self.workflow_root,
            run_root=self.run_root,
            run_inventory={"schema": "run_inventory_v1", "reproducibility_status": "pinned_clean"},
        )

    def test_all_pass_synthetic_fixture_allows_robust_claim(self) -> None:
        status = self.gate_status()

        self.assertTrue(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "robust_allowed")
        self.assertFalse([gate for gate in status["gates"] if gate["status"] == "fail"])

    def test_dirty_repositories_cap_claim_at_diagnostic(self) -> None:
        status = build_gate_status(
            protocol_path=self.protocol_path,
            workflow_root=self.workflow_root,
            run_root=self.run_root,
            run_inventory={"schema": "run_inventory_v1", "reproducibility_status": "pinned_dirty"},
        )

        self.assertFalse(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "diagnostic_only")
        self.assertIn(
            "reproducibility_pinned_clean",
            {gate["id"] for gate in status["gates"] if gate["status"] == "fail"},
        )

    def test_missing_frozen_split_manifest_blocks_robust_claim(self) -> None:
        (self.dataset_root / "frozen_split_manifest.json").unlink()

        status = self.gate_status()

        self.assertFalse(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "invalid_missing_evidence")
        self.assertIn("dataset_joint_a_manifests_present", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})

    def test_missing_final_evaluation_blocks_paper_ready_claim(self) -> None:
        protocol = protocol_payload(self.dataset_root)
        protocol.pop("final_evaluation")
        write_json(self.protocol_path, protocol)

        status = self.gate_status()

        self.assertFalse(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "invalid_protocol")
        self.assertIn("protocol_valid", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})

    def test_deeph_adapter_unproven_blocks_robust_claim(self) -> None:
        adapter = self.run_root / "deeph" / "adapter_manifest.json"
        write_json(
            adapter,
            {
                "schema": "deeph_hdf5_prediction_adapter_v1",
                "robust_matrix_metrics_allowed": False,
                "adapter_equivalence_statuses": ["invalid_orbital_order_unknown"],
                "equivalence_statuses": ["unproven"],
                "equivalence_scopes": ["deeph_processed_blockwise_global_hdf5"],
                "equivalence_gate": {"robust_claim_allowed": False, "diagnostic_only": True},
            },
        )

        status = self.gate_status()

        self.assertFalse(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "invalid_equivalence")
        self.assertIn("deeph_equivalence_proven", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})

    def test_missing_telemetry_blocks_cost_claim(self) -> None:
        stats_path = self.workflow_root / "final_test" / "final_statistics.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["final_seed_summary"][1].pop("gpu_hours_mean")
        write_json(stats_path, stats)

        status = self.gate_status()

        self.assertFalse(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "invalid_telemetry")
        self.assertIn("telemetry_complete", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})

    def test_ml_prediction_reference_path_blocks_dataset_gate(self) -> None:
        split_path = self.dataset_root / "frozen_split_manifest.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["rows"][2]["artifact_paths"]["reference_hsx"] = "ML_prediction.HSX"
        write_json(split_path, split)

        status = self.gate_status()

        self.assertFalse(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "invalid_dataset")
        self.assertIn("forbidden_reference_absent", {gate["id"] for gate in status["gates"] if gate["status"] == "fail"})

    def test_truncated_run_out_blocks_dataset_gate_even_when_manifest_says_valid(self) -> None:
        (self.dataset_root / "s1" / "RUN.out").write_text("SIESTA startup only\n", encoding="utf-8")

        status = self.gate_status()

        self.assertFalse(status["robust_claim_allowed"])
        self.assertEqual(status["claim_status"], "invalid_dataset")
        self.assertIn(
            "dataset_siesta_execution_valid",
            {gate["id"] for gate in status["gates"] if gate["status"] == "fail"},
        )

    def test_malformed_json_returns_nonzero_cli_exit(self) -> None:
        self.protocol_path.write_text("{\n", encoding="utf-8")

        exit_code = main(
            [
                "--protocol",
                str(self.protocol_path),
                "--workflow-root",
                str(self.workflow_root),
                "--run-root",
                str(self.run_root),
                "--output",
                str(self.output_path),
            ]
        )

        self.assertEqual(exit_code, 1)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["robust_claim_allowed"])
        self.assertEqual(payload["claim_status"], "invalid_protocol")


if __name__ == "__main__":
    unittest.main()
