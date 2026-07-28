import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPTS_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402
from graph2mat_sweep_config import (  # noqa: E402
    GRAPH2MAT_EDGE_BLOCK_NODE_MIX,
    GRAPH2MAT_EDGE_MESSAGE_BLOCK,
    GRAPH2MAT_SIMPLE_NODE_BLOCK,
    normalize_graph2mat_overrides,
)
from joint_artifact_contract import validate_dataset  # noqa: E402


def write_snapshot(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")
    (path / "RUN.out").write_text(
        "iscf     Eharris\nSCF cycle converged\nJob completed\n",
        encoding="utf-8",
    )
    (path / "metadata.json").write_text('{"system_label": "graphene"}\n', encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
        if suffix == ".STRUCT_OUT":
            text = "1 0 0\n0 1 0\n0 0 1\n1\n1 6 0 0 0\n"
        elif suffix == ".ORB_INDX":
            text = (
                "      2     2 = orbitals in unit cell and supercell. See end of file.\n\n"
                "io ia is spec iao n l m z p sym rc isc iuo\n"
                "1 1 1 C 1 2 0 0 1 F s 4.0 0 0 0 1\n"
                "2 1 1 C 2 2 1 -1 1 F py 4.0 0 0 0 2\n"
            )
        else:
            text = f"{suffix}\n"
        (path / f"graphene{suffix}").write_text(text, encoding="utf-8")


def write_split_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "split", "sample_dir", "structure_path", "hamiltonian_path", "metadata_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Graph2MatBenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.output_root = self.root / "results"
        self.dataset.mkdir(parents=True)
        (self.dataset / "RUN.fdf").write_text("SystemLabel graphene\nSave.HS T\n", encoding="utf-8")
        (self.dataset / "RUN.out").write_text("SIESTA test log\n", encoding="utf-8")
        (self.dataset / "material_provenance.json").write_text(
            json.dumps(
                {
                    "label": "graphene",
                    "profile": "production",
                    "basis_file_sha256": {"C.ion.xml": "basis"},
                    "pseudopotential_sha256": {"C": "pseudo"},
                    "siesta_version": "SIESTA test-version",
                    "siesta_executable": "siesta",
                    "siesta_command_line": "bash -lc 'siesta < RUN.fdf'",
                    "siesta_stdout_path": str(self.dataset / "RUN.out"),
                    "siesta_returncode": 0,
                    "environment": {"python_version": "3.11.0", "platform": "test-platform"},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def prepare_manifest_dataset(self) -> None:
        split_root = self.dataset / "splits"
        steps = self.dataset / "MD_steps"
        for index, split in enumerate(("train", "validation", "test")):
            step = steps / str(index)
            split_sample = split_root / split / str(index)
            write_snapshot(step)
            write_snapshot(split_sample)
            write_split_manifest(
                split_root / f"{split}_manifest.csv",
                [
                    {
                        "sample_id": f"md_{index}",
                        "split": split,
                        "sample_dir": str(split_sample),
                        "structure_path": str(split_sample / "RUN.fdf"),
                        "hamiltonian_path": str(split_sample / "graphene.TSHS"),
                        "metadata_path": str(split_sample / "metadata.json"),
                    }
                ],
            )
        artifact_validation = validate_dataset(
            steps,
            snapshot_dirs=[steps / "0", steps / "1", steps / "2"],
        ).to_dict()
        (self.dataset / "artifact_validation.json").write_text(
            json.dumps(artifact_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_benchmark_manifests(dataset_root=self.dataset, split_root=split_root)

    def runner_context(self):
        self.prepare_manifest_dataset()
        runner = Graph2MatDeepHBenchmarkRunner()
        validation = runner.validate_dataset_payload({"dataset_root": str(self.dataset)})
        context = runner._prepare_graph2mat_context(
            {
                "dataset_root": str(self.dataset),
                "output_root": str(self.output_root),
                "run_id": "unit_graph2mat",
                "dry_run": True,
            },
            validation,
        )
        return runner, context

    def test_generated_graph2mat_config_points_to_benchmark_splits(self) -> None:
        _, context = self.runner_context()
        config = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["paths"]["dataset_dir"], str(self.dataset))
        self.assertEqual(config["paths"]["training_dir"], str(context.training_dir))
        self.assertEqual(config["training"]["data"]["runs_json"], "runs.json")
        runs_json = json.loads(context.runs_json_path.read_text(encoding="utf-8"))
        self.assertEqual(len(runs_json["train"]), 1)
        self.assertEqual(len(runs_json["val"]), 1)
        self.assertEqual(len(runs_json["test"]), 1)
        self.assertEqual(config["training"]["data"]["matrix_component_policy"], "h_only")
        self.assertEqual(config["training"]["data"]["n_matrix_components"], 1)
        self.assertEqual(
            config["prediction"]["predict_structs"],
            os.path.relpath(str(context.prediction_structs_dir / "*/RUN.fdf"), str(context.training_dir)),
        )
        self.assertFalse(Path(config["training"]["data"]["runs_json"]).is_absolute())
        self.assertEqual(config["prediction"]["output_file"], "ML_prediction.HSX")

    def test_graph2mat_readout_override_is_rendered_to_model_config(self) -> None:
        runner = Graph2MatDeepHBenchmarkRunner()
        config: dict = {"training": {"model": {}, "trainer": {}, "data": {}}}

        runner._apply_graph2mat_overrides(
            config,
            normalize_graph2mat_overrides({"readout": "edge_node_mix"}),
        )

        model = config["training"]["model"]
        self.assertEqual(model["node_block_readout"], GRAPH2MAT_SIMPLE_NODE_BLOCK)
        self.assertEqual(model["edge_block_readout"], GRAPH2MAT_EDGE_BLOCK_NODE_MIX)
        self.assertEqual(model["preprocessing_edges"], GRAPH2MAT_EDGE_MESSAGE_BLOCK)
        self.assertTrue(model["preprocessing_edges_reuse_nodes"])
        self.assertNotIn("readout", model)

    def test_graph2mat_context_materializes_basis_files_next_to_run_fdf(self) -> None:
        basis_dir = self.dataset / "material_basis"
        basis_dir.mkdir(parents=True, exist_ok=True)
        (basis_dir / "C.ion.xml").write_text("<basis />\n", encoding="utf-8")

        _, context = self.runner_context()

        self.assertTrue((self.dataset / "splits" / "train" / "0" / "C.ion.xml").exists())
        self.assertTrue((context.prediction_structs_dir / "md_2" / "C.ion.xml").exists())

    def test_graph2mat_prediction_scripts_do_not_restore_loaded_checkpoint_twice(self) -> None:
        prediction_text = (REPO_ROOT / "MD" / "scripts" / "run_md_prediction.py").read_text(
            encoding="utf-8"
        )
        testing_text = (REPO_ROOT / "MD" / "scripts" / "run_md_testing.py").read_text(
            encoding="utf-8"
        )
        cross_prediction_text = (
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py"
        ).read_text(encoding="utf-8")

        for text in (prediction_text, testing_text, cross_prediction_text):
            self.assertIn("weights_only=False", text)
            self.assertIn("ckpt_path=None", text)

    def test_missing_validation_split_fails(self) -> None:
        steps = self.dataset / "MD_steps"
        write_snapshot(steps / "0")
        split_root = self.dataset / "splits"
        train_sample = split_root / "train" / "0"
        test_sample = split_root / "test" / "0"
        write_snapshot(train_sample)
        write_snapshot(test_sample)
        (self.dataset / "artifact_validation.json").write_text(
            json.dumps(validate_dataset(steps, snapshot_dirs=[steps / "0"]).to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        (self.dataset / "benchmark_dataset_manifest.json").write_text(
            json.dumps({"benchmark_ready": True}, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.dataset / "frozen_split_manifest.json").write_text(
            json.dumps(
                {
                    "valid": True,
                    "split_hash": "abc",
                    "split_counts": {"train": 1, "validation": 0, "test": 1},
                    "rows": [
                        {"sample_id": "train0", "split": "train", "sample_dir": str(train_sample)},
                        {"sample_id": "test0", "split": "test", "sample_dir": str(test_sample)},
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        runner = Graph2MatDeepHBenchmarkRunner()
        validation = runner.validate_dataset_payload({"dataset_root": str(self.dataset)})
        with self.assertRaisesRegex(RuntimeError, "validation"):
            runner._prepare_graph2mat_context(
                {
                    "dataset_root": str(self.dataset),
                    "output_root": str(self.output_root),
                    "run_id": "missing_validation",
                    "dry_run": True,
                },
                validation,
            )

    def test_prediction_outputs_expected_for_every_test_sample(self) -> None:
        runner, context = self.runner_context()
        for structure in Path(context.prediction_structs_dir).glob("*/RUN.fdf"):
            (structure.parent / "ML_prediction.HSX").write_text("prediction\n", encoding="utf-8")

        outputs = runner._validate_graph2mat_prediction_outputs(context)

        self.assertEqual(outputs["count"], 1)
        self.assertTrue(outputs["rows"][0]["sha256"])

    def test_checkpoint_provenance_is_written(self) -> None:
        runner, context = self.runner_context()
        checkpoint_manifest = {
            "checkpoint_path": str(context.training_dir / "best.ckpt"),
            "checkpoint_sha256": "ckpt-hash",
            "selection_metric": "val_loss",
        }
        manifest = runner._write_graph2mat_manifest(
            context,
            checkpoint_manifest=checkpoint_manifest,
            extra={"training_completed": True},
        )

        written = json.loads(context.graph2mat_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["checkpoint_manifest"]["checkpoint_sha256"], "ckpt-hash")
        self.assertEqual(written["checkpoint_manifest"]["selection_metric"], "val_loss")
        self.assertEqual(written["context"]["split_hash"], context.split_hash)

    def test_dry_run_workflow_completes_graph2mat_stage_without_training(self) -> None:
        self.prepare_manifest_dataset()
        runner = Graph2MatDeepHBenchmarkRunner()
        status = runner.start(
            {
                "dataset_root": str(self.dataset),
                "output_root": str(self.output_root),
                "run_id": "dry_run_workflow",
                "dry_run": True,
            }
        )
        self.assertTrue(status["running"])
        runner._thread.join(timeout=5)

        final_status = runner.status()
        self.assertFalse(final_status["running"])
        self.assertEqual(final_status["returncode"], 0)
        self.assertEqual(final_status["stage"], "complete")
        self.assertTrue((self.output_root / "dry_run_workflow" / "graph2mat" / "pipeline_config.yaml").exists())
        self.assertTrue(
            (self.output_root / "dry_run_workflow" / "graph2mat" / "graph2mat_manifest.json").exists()
        )

    def test_runner_module_does_not_depend_on_experiment_runner_state(self) -> None:
        source = (SCRIPTS_DIR / "g2m_deeph_runner.py").read_text(encoding="utf-8")
        self.assertNotIn("ExperimentRunner", source)
        self.assertNotIn("EXPERIMENT_RUNNER", source)


if __name__ == "__main__":
    unittest.main()
