from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MD_SCRIPTS_DIR = REPO_ROOT / "MD" / "scripts"
if str(MD_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(MD_SCRIPTS_DIR))


def load_md_config_module():
    spec = importlib.util.spec_from_file_location(
        "md_pipeline_config_joint_artifacts_test",
        MD_SCRIPTS_DIR / "md_pipeline_config.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_generate_md_dataset_module():
    spec = importlib.util.spec_from_file_location(
        "generate_md_dataset_joint_artifacts_test",
        MD_SCRIPTS_DIR / "generate_md_dataset.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_joint_snapshot(
    path: Path,
    *,
    label: str = "graphene",
    include_hsx: bool = True,
    include_struct_out: bool = True,
    include_orb_indx: bool = True,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text(f"SystemLabel {label}\n", encoding="utf-8")
    (path / "metadata.json").write_text('{"system_label": "%s"}\n' % label, encoding="utf-8")
    (path / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".XV"):
        (path / f"{label}{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
    if include_hsx:
        (path / f"{label}.HSX").write_text("hsx\n", encoding="utf-8")
    if include_struct_out:
        (path / f"{label}.STRUCT_OUT").write_text("struct\n", encoding="utf-8")
    if include_orb_indx:
        (path / f"{label}.ORB_INDX").write_text("orb\n", encoding="utf-8")


class MDJointArtifactGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.md_config = load_md_config_module()
        self.generate = load_generate_md_dataset_module()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def config(self) -> dict:
        config = self.md_config.load_pipeline_config(REPO_ROOT / "MD" / "pipeline_config.yaml")
        config["paths"]["dataset_dir"] = str(self.root / "dataset")
        config["paths"]["training_dir"] = str(self.root / "training")
        dataset = Path(config["paths"]["dataset_dir"])
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "RUN.fdf").write_text("SystemLabel graphene\nSave.HS T\n", encoding="utf-8")
        (dataset / "material_provenance.json").write_text(
            json.dumps(
                {
                    "label": "graphene",
                    "basis_file_sha256": {"C.ion.xml": "basis"},
                    "pseudopotential_sha256": {"C": "pseudo"},
                    "fdf_sha256": "fdfhash",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return config

    def test_rendered_fdf_contains_deeph_required_output_flags(self) -> None:
        text = self.md_config.render_run_fdf(self.config())

        for pattern in (
            r"(?m)^SaveHS\s+true\b",
            r"(?m)^Save\.HS\s+T\b",
            r"(?m)^TS\.HS\.Save\s+T\b",
            r"(?m)^TS\.DE\.Save\s+T\b",
            r"(?m)^XML\.Write\s+T\b",
            r"(?m)^Write\.OrbitalIndex\s+T\b",
        ):
            self.assertRegex(text, pattern)

    def test_setup_store_command_uses_expanded_joint_store_files(self) -> None:
        config = self.config()

        with mock.patch.object(self.generate, "run_command") as run_command:
            self.generate.setup_store(config)

        command = run_command.call_args.args[0]
        self.assertIn("--files", command)
        store_files = command[command.index("--files") + 1]
        for pattern in ("*fdf", "*TSHS", "*TSDE", "*XV", "*HSX", "*STRUCT_OUT", "*ORB_INDX", "*out"):
            self.assertIn(pattern, store_files)

    def test_joint_metadata_is_created_for_generated_benchmark_snapshot(self) -> None:
        config = self.config()
        sample = Path(config["paths"]["dataset_dir"]) / "MD_steps" / "0"
        write_joint_snapshot(sample)
        (sample / "metadata.json").unlink()

        metadata = self.generate.write_joint_snapshot_metadata(
            sample,
            config,
            extra={"temperature_K": 300, "source_frame_index": "0"},
        )

        self.assertTrue((sample / "metadata.json").exists())
        self.assertEqual(metadata["artifact_contract_version"], "joint_graph2mat_deeph_artifact_contract_v1")
        self.assertEqual(metadata["generation_mode"], "clean_one_pass")
        self.assertEqual(metadata["source"], "graph2mat_vs_deeph_dataset_generation")
        self.assertTrue(metadata["artifacts"]["hsx"]["present"])
        self.assertTrue(metadata["artifacts"]["struct_out"]["present"])
        self.assertTrue(metadata["artifacts"]["orb_indx"]["present"])
        self.assertIn("*HSX", metadata["joint_store_file_patterns"])

    def test_graph2mat_basis_is_materialized_into_samples_and_splits(self) -> None:
        config = self.config()
        dataset = Path(config["paths"]["dataset_dir"])
        basis_dir = dataset / "material_basis"
        basis_dir.mkdir(parents=True)
        (basis_dir / "C.ion.xml").write_text("<basis />\n", encoding="utf-8")
        sample = dataset / "MD_steps" / "0"
        write_joint_snapshot(sample)

        materialized = self.generate._materialize_graph2mat_basis_files(dataset, [sample])
        self.assertEqual(materialized, 1)
        self.assertTrue((sample / "C.ion.xml").exists())

        split_sample = dataset / "splits" / "train" / "0"
        self.generate._prepare_split_sample(sample, split_sample)
        self.assertTrue((split_sample / "C.ion.xml").exists())

    def test_old_default_graph2mat_store_list_is_not_used(self) -> None:
        config = self.config()

        with mock.patch.object(self.generate, "run_command") as run_command:
            self.generate.setup_store(config)

        command = run_command.call_args.args[0]
        self.assertNotEqual(command, [config["commands"]["graph2mat"], "siesta", "md", "setup-store"])
        self.assertNotEqual(
            command[command.index("--files") + 1],
            "*fdf *TSHS *TSDE *XV",
        )

    def test_temperature_block_copy_includes_material_provenance(self) -> None:
        config = self.config()
        dataset = Path(config["paths"]["dataset_dir"])
        block_dir = self.root / "block"

        self.generate._copy_pseudopotentials_for_block(dataset, block_dir)

        copied = block_dir / "material_provenance.json"
        self.assertTrue(copied.exists())
        self.assertEqual(
            json.loads(copied.read_text(encoding="utf-8"))["pseudopotential_sha256"],
            {"C": "pseudo"},
        )

    def test_block_validation_passes_after_material_provenance_copy(self) -> None:
        config = self.config()
        dataset = Path(config["paths"]["dataset_dir"])
        block_dir = self.root / "block"
        block_dir.mkdir(parents=True)
        self.generate._copy_pseudopotentials_for_block(dataset, block_dir)
        (block_dir / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")
        block_config = self.generate._block_config(config, block_dir, {"n_snapshots": 1})
        write_joint_snapshot(block_dir / "MD_steps" / "0")

        self.generate.validate_joint_benchmark_artifacts(block_config)

        summary = json.loads((block_dir / "artifact_validation.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["benchmark_ready"])

    def test_dataset_level_provenance_errors_are_reported(self) -> None:
        config = self.config()
        dataset = Path(config["paths"]["dataset_dir"])
        (dataset / "material_provenance.json").unlink()
        steps_dir = dataset / "MD_steps"
        write_joint_snapshot(steps_dir / "0")

        with self.assertRaisesRegex(RuntimeError, "Dataset-level errors: .*pseudopotential"):
            self.generate.validate_joint_benchmark_artifacts(config)

    def test_missing_archived_hsx_fails_joint_validation(self) -> None:
        config = self.config()
        steps_dir = Path(config["paths"]["dataset_dir"]) / "MD_steps"
        write_joint_snapshot(steps_dir / "0", include_hsx=False)

        with self.assertRaisesRegex(RuntimeError, "joint_graph2mat_deeph_artifact_contract_v1"):
            self.generate.validate_joint_benchmark_artifacts(config)
        summary = Path(config["paths"]["dataset_dir"]) / "artifact_validation.json"
        self.assertTrue(summary.exists())
        self.assertIn('"repair_required"', summary.read_text(encoding="utf-8"))

    def test_missing_archived_orb_indx_fails_joint_validation(self) -> None:
        config = self.config()
        steps_dir = Path(config["paths"]["dataset_dir"]) / "MD_steps"
        write_joint_snapshot(steps_dir / "0", include_orb_indx=False)

        with self.assertRaisesRegex(RuntimeError, "orb_indx"):
            self.generate.validate_joint_benchmark_artifacts(config)

    def test_complete_archived_snapshot_passes_joint_validation(self) -> None:
        config = self.config()
        steps_dir = Path(config["paths"]["dataset_dir"]) / "MD_steps"
        write_joint_snapshot(steps_dir / "0")

        self.generate.validate_joint_benchmark_artifacts(config)
        summary = Path(config["paths"]["dataset_dir"]) / "artifact_validation.json"
        self.assertTrue(summary.exists())
        text = summary.read_text(encoding="utf-8")
        self.assertIn('"benchmark_ready": true', text)
        self.assertIn('"valid_snapshots": 1', text)
        metadata = json.loads((steps_dir / "0" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["artifact_contract_validation_status"], "valid")

    def test_old_graph2mat_only_dataset_writes_repair_required_summary(self) -> None:
        config = self.config()
        steps_dir = Path(config["paths"]["dataset_dir"]) / "MD_steps"
        write_joint_snapshot(
            steps_dir / "0",
            include_hsx=False,
            include_struct_out=False,
            include_orb_indx=False,
        )

        with self.assertRaisesRegex(RuntimeError, "hsx, struct_out, orb_indx"):
            self.generate.validate_joint_benchmark_artifacts(config)

        summary = (Path(config["paths"]["dataset_dir"]) / "artifact_validation.json").read_text(
            encoding="utf-8"
        )
        self.assertIn('"scientific_status": "repair_required"', summary)
        self.assertIn('"repair_required_snapshots": 1', summary)

    def test_invalid_snapshots_do_not_create_splits(self) -> None:
        config = self.config()
        steps_dir = Path(config["paths"]["dataset_dir"]) / "MD_steps"
        write_joint_snapshot(steps_dir / "0", include_hsx=False)

        with self.assertRaises(RuntimeError):
            self.generate.validate_joint_benchmark_artifacts(config)

        self.assertFalse((Path(config["paths"]["dataset_dir"]) / "splits").exists())


if __name__ == "__main__":
    unittest.main()
