from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Comparison" / "scripts" / "export_graph2mat_matrix_error_plot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_graph2mat_matrix_error_plot_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExportGraph2MatMatrixErrorPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / "best.ckpt"
        self.ckpt.write_text("checkpoint\n", encoding="utf-8")
        self.run = self.root / "dataset" / "md_0" / "RUN.fdf"
        self.run.parent.mkdir(parents=True)
        self.run.write_text("SystemLabel graphene\n", encoding="utf-8")
        self.output = self.root / "out"
        self.module = load_module()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def parse(self, *args: str):
        return self.module.parse_args(
            [
                "--ckpt-path",
                str(self.ckpt),
                "--test-runs",
                str(self.run),
                "--output-dir",
                str(self.output),
                *args,
            ]
        )

    def test_build_graph2mat_test_command(self) -> None:
        args = self.parse("--show", "--store-in-logger", "--out-matrix", "hamiltonian")
        command = self.module.build_graph2mat_test_command(
            args,
            sample_metrics_csv=self.output / "sample_metrics.csv",
        ).command

        self.assertEqual(command[:5], ["graph2mat", "models", "mace", "main", "test"])
        self.assertIn("--ckpt_path", command)
        self.assertIn(str(self.ckpt), command)
        self.assertIn("--data.test_runs", command)
        self.assertIn("--data.out_matrix", command)
        self.assertIn("hamiltonian", command)
        self.assertIn("--trainer.callbacks+", command)
        self.assertIn("PlotMatrixError", command)
        self.assertIn("SamplewiseMetricsLogger", command)
        self.assertIn("--trainer.callbacks.show", command)
        self.assertIn("true", command)

    def test_manifest_contains_required_fields(self) -> None:
        args = self.parse("--dry-run")
        payload = self.module.run(args)
        manifest_path = self.output / "matrix_error_manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "planned_dry_run")
        self.assertEqual(manifest["ckpt_path"], str(self.ckpt))
        self.assertEqual(manifest["split"], "test")
        self.assertEqual(manifest["output_dir"], str(self.output))
        self.assertIn("outputs", manifest)
        self.assertIn("input_hashes", manifest)
        self.assertEqual(manifest["status"], "planned_dry_run")

    def test_reject_missing_checkpoint(self) -> None:
        missing = self.root / "missing.ckpt"
        args = self.module.parse_args(
            [
                "--ckpt-path",
                str(missing),
                "--test-runs",
                str(self.run),
                "--output-dir",
                str(self.output),
                "--dry-run",
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "Checkpoint not found"):
            self.module.run(args)

    def test_reject_missing_test_runs(self) -> None:
        args = self.module.parse_args(
            [
                "--ckpt-path",
                str(self.ckpt),
                "--test-runs",
                str(self.root / "missing" / "*" / "RUN.fdf"),
                "--output-dir",
                str(self.output),
                "--dry-run",
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "--test-runs did not match"):
            self.module.run(args)

    def test_backend_headless_safe_by_default(self) -> None:
        args = self.parse("--dry-run")
        self.assertFalse(args.show)
        payload = self.module.run(args)
        command = payload["command"]
        show_index = command.index("--trainer.callbacks.show")
        self.assertEqual(command[show_index + 1], "false")

    def test_save_flags_are_reflected_in_manifest(self) -> None:
        average_root = self.root / "predictions"
        average_root.mkdir()
        args = self.parse(
            "--mode", "programmatic", "--dry-run", "--error-metric", "mae",
            "--average-hsx-root", str(average_root),
            "--crop-empty", "--save-pdf", "--no-save-html", "--no-save-png",
        )
        self.module.run(args)
        manifest = json.loads((self.output / "matrix_error_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["mode"], "programmatic")
        self.assertEqual(manifest["save_flags"], {"html": False, "pdf": True, "png": False})
        self.assertEqual(manifest["error_metric"], "mae")
        self.assertTrue(manifest["crop_empty"])
        self.assertEqual(manifest["aggregation_mode"], "dataset_mean")
        self.assertEqual(manifest["status"], "planned_dry_run")

    def test_glob_test_runs_are_resolved(self) -> None:
        args = self.module.parse_args(
            [
                "--ckpt-path",
                str(self.ckpt),
                "--test-runs",
                str(self.root / "dataset" / "*" / "RUN.fdf"),
                "--output-dir",
                str(self.output),
                "--dry-run",
            ]
        )
        runs = self.module.validate_inputs(args)

        self.assertEqual(runs, [self.run])

    def test_programmatic_paths_are_resolved_from_config_directory(self) -> None:
        config_dir = self.root / "run" / "graph2mat"
        config_dir.mkdir(parents=True)
        config = config_dir / "pipeline_config.yaml"
        config.write_text(
            "training:\n  data:\n    basis_files: ../dataset/*.ion.xml\n    root_dir: .\n",
            encoding="utf-8",
        )
        args = self.parse("--mode", "programmatic", "--config-yaml", str(config), "--dry-run")

        kwargs = self.module.programmatic_datamodule_kwargs(args, self.module.load_yaml_config(config))

        self.assertEqual(kwargs["basis_files"], "../dataset/*.ion.xml")
        self.assertEqual(kwargs["root_dir"], str(config_dir.resolve()))

    def test_matrix_plot_ranges_crop_empty_supercell_blocks(self) -> None:
        matrix = csr_matrix(np.pad(np.ones((2, 2)), ((0, 0), (0, 4))))
        matrix[0, 4] = 1

        crop_x, crop_y, full_x, full_y = self.module.matrix_plot_ranges(matrix)

        self.assertEqual(crop_x, [-0.5, 1.5])
        self.assertEqual(crop_y, [1.5, -0.5])
        self.assertEqual(full_x, [-0.5, 5.5])
        self.assertEqual(full_y, [1.5, -0.5])


if __name__ == "__main__":
    unittest.main()
