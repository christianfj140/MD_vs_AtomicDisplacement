import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPTS_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from g2m_deeph_final_workflow import main as workflow_main  # noqa: E402
from g2m_deeph_verify_protocol_datasets import main, verify_protocol_datasets  # noqa: E402
from joint_artifact_contract import validate_dataset  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_snapshot(path: Path, *, label: str = "graphene") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text(
        "\n".join(
            [
                f"SystemLabel {label}",
                "SaveHS true",
                "Save.HS T",
                "TS.HS.Save T",
                "TS.DE.Save T",
                "XML.Write T",
                "Write.OrbitalIndex T",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (path / "RUN.out").write_text("Job completed\n", encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps({"system_label": label}) + "\n", encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
        (path / f"{label}{suffix}").write_text(f"{suffix}\n", encoding="utf-8")


def write_split_csv(path: Path, sample_id: str, sample_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "sample_dir"])
        writer.writeheader()
        writer.writerow({"sample_id": sample_id, "split": path.stem.replace("_manifest", ""), "sample_dir": str(sample_dir)})


def protocol_payload(dataset_root: Path) -> dict:
    return {
        "protocol_id": "dataset_verify_protocol_unit",
        "version": "1.0",
        "datasets": [
            {
                "dataset_id": "joint_a",
                "dataset_root": str(dataset_root),
                "benchmark_dataset_manifest": str(dataset_root / "benchmark_dataset_manifest.json"),
                "frozen_split_manifest": str(dataset_root / "frozen_split_manifest.json"),
                "split_root": str(dataset_root / "splits"),
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
                    "optim_lr": [0.001, 0.003],
                    "batch_size": [64, 128],
                    "max_epochs": [20],
                    "hidden_irreps": ["16x0e + 16x1o + 16x2e + 16x3o"],
                    "num_interactions": [2],
                    "correlation": [2],
                    "max_ell": [2],
                },
            },
            "deeph": {
                "enabled": True,
                "search_space": {
                    "learning_rate": [0.0001, 0.0003],
                    "batch_size": [2, 4],
                    "epochs": [20],
                    "atom_fea_len": [64],
                    "edge_fea_len": [128],
                    "num_l": [4],
                    "if_lcmp": [True],
                },
            },
        },
        "selection": {"split": "validation", "metric": "val_loss", "mode": "min", "source": "validation_only"},
        "early_stopping": {"metric": "val_loss", "mode": "min", "patience": 5, "min_delta": 0.0, "max_epochs": 20},
        "search_policy": {"strategy": "random", "n_trials_per_model": 2, "random_seed": 1},
        "budget_policy": {"mode": "equal_n_trials", "n_trials_per_model": 2},
        "final_seeds": [0, 1, 2],
        "top_k_selection": {"k_per_model": 1, "split": "validation", "metric": "val_loss", "uses_test_metrics": False},
        "final_evaluation": {"primary_metric": "low_energy_rmse_eV", "mode": "min", "secondary_metrics": ["h_mae_eV"]},
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


class Graph2MatDeepHVerifyProtocolDatasetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.protocol_path = self.root / "protocol.json"
        self.output_path = self.root / "dataset_verification.json"
        self.sample_dirs = self.create_dataset()
        write_json(self.protocol_path, protocol_payload(self.dataset))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def create_dataset(self, *, write_manifests: bool = True) -> list[Path]:
        self.dataset.mkdir(parents=True, exist_ok=True)
        (self.dataset / "RUN.out").write_text("Job completed\n", encoding="utf-8")
        write_json(
            self.dataset / "material_provenance.json",
            {
                "label": "graphene",
                "fdf_sha256": "fdfhash",
                "basis_file_sha256": {"C.ion.xml": "basis"},
                "pseudopotential_sha256": {"C": "pseudo"},
                "siesta_version": "SIESTA 5.4.2-unit-test",
                "siesta_command_line": "siesta < RUN.fdf",
                "run_out_path": str(self.dataset / "RUN.out"),
                "environment": {"python_version": "3.11", "platform": "test"},
            },
        )
        sample_dirs: list[Path] = []
        for index, split in enumerate(("train", "validation", "test")):
            sample_dir = self.dataset / "samples" / f"s{index}"
            write_snapshot(sample_dir)
            sample_dirs.append(sample_dir)
            write_split_csv(self.dataset / "splits" / f"{split}_manifest.csv", f"s{index}", sample_dir)
        artifact_validation = validate_dataset(self.dataset, snapshot_dirs=sample_dirs).to_dict()
        write_json(self.dataset / "artifact_validation.json", artifact_validation)
        if write_manifests:
            write_benchmark_manifests(
                dataset_root=self.dataset,
                split_root=self.dataset / "splits",
                strict_paper_ready_provenance=True,
            )
        return sample_dirs

    def test_valid_minimal_dataset_fixture_passes(self) -> None:
        payload = verify_protocol_datasets(protocol_path=self.protocol_path, output_path=self.output_path, strict=True)

        self.assertEqual(payload["status"], "valid")
        self.assertEqual(payload["datasets"][0]["status"], "valid")
        self.assertTrue(self.output_path.exists())

    def test_missing_frozen_split_fails(self) -> None:
        (self.dataset / "frozen_split_manifest.json").unlink()

        payload = verify_protocol_datasets(protocol_path=self.protocol_path, output_path=self.output_path, strict=True)

        self.assertEqual(payload["status"], "invalid")
        self.assertTrue(any("frozen_split_manifest" in blocker for blocker in payload["blockers"]))

    def test_invalid_split_hash_link_fails(self) -> None:
        benchmark_path = self.dataset / "benchmark_dataset_manifest.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        benchmark["frozen_split_manifest"]["split_hash"] = "wrong-hash"
        write_json(benchmark_path, benchmark)

        payload = verify_protocol_datasets(protocol_path=self.protocol_path, output_path=self.output_path, strict=True)

        self.assertEqual(payload["status"], "invalid")
        self.assertTrue(any("split_hash mismatch" in blocker for blocker in payload["blockers"]))

    def test_missing_provenance_fails_in_strict_mode(self) -> None:
        material_path = self.dataset / "material_provenance.json"
        material = json.loads(material_path.read_text(encoding="utf-8"))
        material.pop("siesta_version")
        write_json(material_path, material)

        payload = verify_protocol_datasets(protocol_path=self.protocol_path, output_path=self.output_path, strict=True)

        self.assertEqual(payload["status"], "invalid")
        self.assertTrue(any("SIESTA version provenance" in blocker for blocker in payload["blockers"]))

    def test_write_manifests_writes_expected_files_when_split_root_exists(self) -> None:
        (self.dataset / "benchmark_dataset_manifest.json").unlink()
        (self.dataset / "frozen_split_manifest.json").unlink()

        payload = verify_protocol_datasets(
            protocol_path=self.protocol_path,
            output_path=self.output_path,
            strict=True,
            write_manifests=True,
        )

        self.assertEqual(payload["status"], "valid")
        self.assertTrue((self.dataset / "benchmark_dataset_manifest.json").exists())
        self.assertTrue((self.dataset / "frozen_split_manifest.json").exists())

    def test_cli_returns_nonzero_for_invalid_dataset(self) -> None:
        (self.dataset / "frozen_split_manifest.json").unlink()

        exit_code = main(["--protocol", str(self.protocol_path), "--output", str(self.output_path), "--strict"])

        self.assertEqual(exit_code, 1)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "invalid")

    def test_final_workflow_validate_protocol_can_verify_datasets(self) -> None:
        workflow = self.root / "workflow"

        exit_code = workflow_main(
            [
                "--stage",
                "validate-protocol",
                "--protocol",
                str(self.protocol_path),
                "--workflow-root",
                str(workflow),
                "--verify-datasets",
            ]
        )

        self.assertEqual(exit_code, 0)
        verification = json.loads((workflow / "dataset_verification.json").read_text(encoding="utf-8"))
        stage = json.loads((workflow / "stages" / "validate-protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(verification["status"], "valid")
        self.assertEqual(stage["outputs"]["dataset_verification_status"], "valid")


if __name__ == "__main__":
    unittest.main()
