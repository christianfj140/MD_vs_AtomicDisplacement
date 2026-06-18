import json
import csv
import importlib.util
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from contextlib import ExitStack
from unittest import mock
from http.server import ThreadingHTTPServer
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_runner import (  # noqa: E402
    CommandRunError,
    DEFAULT_DATASET_ROOT,
    DeepHBenchmarkContext,
    Graph2MatBenchmarkContext,
    METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
    METRIC_FAIL_POLICY_FAIL_CLOSED,
    Graph2MatDeepHBenchmarkRunner,
    backfill_derivative_postprocess_from_training_sweep,
    build_derivative_model_comparison_summary,
    _deeph_device_settings,
    _deeph_training_parallelism,
    _deeph_metric_command_args,
    _derivative_metric_command_args,
    _derivative_metrics_settings,
    _extract_validation_metrics,
    _force_diagnostic_metric_manifest,
    _link_or_copy_file,
    _metric_allowed_returncodes,
    _metric_evaluation_split,
    _metric_fail_policy,
    _normalized_modular_workflow_payload,
    _write_csv,
)


def _write_fake_deeph_cli(repo_or_bin: Path, command_name: str) -> Path:
    bin_dir = repo_or_bin / ".venv" / "bin" if repo_or_bin.name != "bin" else repo_or_bin
    bin_dir.mkdir(parents=True, exist_ok=True)
    command = bin_dir / command_name
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    return command


def _write_snapshot(root: Path, sample_id: str, *, complete: bool) -> Path:
    snapshot = root / "MD_steps" / sample_id
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")
    (snapshot / "RUN.out").write_text("Job completed\n", encoding="utf-8")
    (snapshot / "metadata.json").write_text('{"system_label": "graphene"}\n', encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".XV"):
        (snapshot / f"graphene{suffix}").write_text("artifact\n", encoding="utf-8")
    if complete:
        (snapshot / "graphene.HSX").write_text("artifact\n", encoding="utf-8")
        (snapshot / "graphene.STRUCT_OUT").write_text(
            "1 0 0\n0 1 0\n0 0 1\n1\n1 6 0 0 0\n",
            encoding="utf-8",
        )
        (snapshot / "graphene.ORB_INDX").write_text(
            "      2     2 = orbitals in unit cell and supercell. See end of file.\n\n"
            "io ia is spec iao n l m z p sym rc isc iuo\n"
            "1 1 1 C 1 2 0 0 1 F s 4.0 0 0 0 1\n"
            "2 1 1 C 2 2 1 -1 1 F py 4.0 0 0 0 2\n",
            encoding="utf-8",
        )
    return snapshot


def _write_dataset_provenance(dataset: Path) -> None:
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "RUN.fdf").write_text("SystemLabel graphene\nSave.HS T\n", encoding="utf-8")
    (dataset / "RUN.out").write_text("SIESTA test log\n", encoding="utf-8")
    (dataset / "material_provenance.json").write_text(
        json.dumps(
            {
                "label": "graphene",
                "basis_file_sha256": {"C.ion.xml": "basis"},
                "pseudopotential_sha256": {"C": "pseudo"},
                "fdf_sha256": "fdfhash",
                "siesta_version": "SIESTA test-version",
                "siesta_executable": "siesta",
                "siesta_command_line": "bash -lc 'siesta < RUN.fdf'",
                "siesta_stdout_path": str(dataset / "RUN.out"),
                "siesta_returncode": 0,
                "environment": {"python_version": "3.11.0", "platform": "test-platform"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fake_joint_dataset_from_config(config_path: Path, *, complete: bool = True) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = Path(config["paths"]["dataset_dir"])
    _write_dataset_provenance(dataset)
    steps = int(config["md"]["steps"])
    for index in range(steps):
        _write_snapshot(dataset, str(index), complete=complete)
    for split, count in (
        ("train", int(config["splits"]["train"])),
        ("validation", int(config["splits"]["validation"])),
        ("test", int(config["splits"]["test"])),
    ):
        manifest = dataset / "splits" / f"{split}_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        rows = ["sample_id,split,sample_dir\n"]
        for index in range(count):
            sample = dataset / "MD_steps" / str(index)
            target = dataset / "splits" / split / str(index)
            target.mkdir(parents=True, exist_ok=True)
            for src in sample.iterdir():
                if src.is_file():
                    (target / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            rows.append(f"md_{index},{split},{target}\n")
        manifest.write_text("".join(rows), encoding="utf-8")
    (dataset / "artifact_validation.json").write_text('{"valid": true, "snapshots": []}\n', encoding="utf-8")
    (dataset / "benchmark_dataset_manifest.json").write_text('{"benchmark_ready": true}\n', encoding="utf-8")
    (dataset / "frozen_split_manifest.json").write_text('{"valid": true, "rows": []}\n', encoding="utf-8")


def _write_training_ready_dataset(dataset: Path) -> None:
    _write_dataset_provenance(dataset)
    steps = dataset / "MD_steps"
    split_root = dataset / "splits"
    rows = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for index, split in enumerate(("train", "validation", "test")):
        sample_id = f"md_{index}"
        source = _write_snapshot(dataset, str(index), complete=True)
        target = split_root / split / sample_id
        target.mkdir(parents=True, exist_ok=True)
        for src in source.iterdir():
            if src.is_file():
                (target / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = split_root / f"{split}_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "sample_id,split,sample_dir,structure_path,hamiltonian_path,metadata_path\n"
            f"{sample_id},{split},{target},{target / 'RUN.fdf'},{target / 'graphene.TSHS'},{target / 'metadata.json'}\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "sample_dir": str(target),
                "structure_path": str(target / "RUN.fdf"),
                "hamiltonian_path": str(target / "graphene.TSHS"),
                "metadata_path": str(target / "metadata.json"),
            }
        )
        split_counts[split] += 1
    (dataset / "artifact_validation.json").write_text('{"valid": true, "snapshots": []}\n', encoding="utf-8")
    (dataset / "benchmark_dataset_manifest.json").write_text(
        json.dumps({"benchmark_ready": True, "dataset_root": str(dataset)}) + "\n",
        encoding="utf-8",
    )
    (dataset / "frozen_split_manifest.json").write_text(
        json.dumps({"valid": True, "split_hash": "unit-split", "split_counts": split_counts, "rows": rows}) + "\n",
        encoding="utf-8",
    )


class Graph2MatDeepHRunnerTests(unittest.TestCase):
    def test_status_before_run_is_idle(self):
        runner = Graph2MatDeepHBenchmarkRunner()
        status = runner.status()
        self.assertFalse(status["running"])
        self.assertEqual(status["stage"], "idle")
        self.assertEqual(status["contract_name"], "joint_graph2mat_deeph_artifact_contract_v1")

    def test_metric_fail_policy_defaults_to_production_fail_closed(self):
        self.assertEqual(_metric_fail_policy({}), METRIC_FAIL_POLICY_FAIL_CLOSED)
        self.assertEqual(_metric_allowed_returncodes(METRIC_FAIL_POLICY_FAIL_CLOSED), (0,))
        command = _deeph_metric_command_args(
            python_executable="python3",
            graph2mat_result_dir=Path("g2m"),
            processed_dir=Path("processed"),
            predictions_dir=Path("predictions"),
            output_dir=Path("out"),
            metric_fail_policy=METRIC_FAIL_POLICY_FAIL_CLOSED,
        )

        self.assertNotIn("--no-fail-closed", command)
        self.assertIn("test", command)

    def test_metric_fail_policy_diagnostic_mode_is_explicit(self):
        self.assertEqual(
            _metric_fail_policy({"allow_diagnostic_metrics": True}),
            METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
        )
        self.assertEqual(_metric_allowed_returncodes(METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY), (0, 2))
        command = _deeph_metric_command_args(
            python_executable="python3",
            graph2mat_result_dir=Path("g2m"),
            processed_dir=Path("processed"),
            predictions_dir=Path("predictions"),
            output_dir=Path("out"),
            metric_fail_policy=METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
        )

        self.assertIn("--no-fail-closed", command)

    def test_derivative_metrics_payload_defaults_disabled(self):
        settings = _derivative_metrics_settings({})

        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["finite_difference_method"], "central")
        self.assertEqual(settings["split"], "test")
        self.assertTrue(settings["require_central"])
        self.assertTrue(settings["diagnostic_only"])
        self.assertEqual(settings["support_threshold"], 1e-12)

    def test_derivative_metric_command_args_include_expected_cli_options(self):
        settings = _derivative_metrics_settings(
            {
                "derivative_metrics": {
                    "enabled": True,
                    "finite_difference_method": "central",
                    "split": "test",
                    "require_central": True,
                    "diagnostic_only": True,
                    "support_threshold": 1e-10,
                    "max_stencils": 3,
                }
            }
        )

        command = _derivative_metric_command_args(
            python_executable="/venv/bin/python",
            result_dir=Path("/tmp/result"),
            output_dir=Path("/tmp/result/derivative_metrics"),
            source_model="graph2mat",
            settings=settings,
        )

        self.assertIn("evaluate_hamiltonian_derivative_metrics.py", command[1])
        self.assertIn("--require-central", command)
        self.assertIn("--diagnostic-only", command)
        self.assertIn("--overwrite", command)
        self.assertEqual(command[command.index("--source-model") + 1], "graph2mat")
        self.assertEqual(command[command.index("--max-stencils") + 1], "3")

    def test_modular_workflow_defaults_preserve_existing_hamiltonian_path(self):
        workflow = _normalized_modular_workflow_payload({})
        stages = workflow["stages"]

        self.assertEqual(workflow["workflow_mode"], "default")
        self.assertTrue(stages["generate_or_validate_dataset"])
        self.assertTrue(stages["freeze_splits"])
        self.assertTrue(stages["train_graph2mat"])
        self.assertTrue(stages["predict_graph2mat"])
        self.assertTrue(stages["train_deeph"])
        self.assertTrue(stages["predict_deeph"])
        self.assertTrue(stages["hamiltonian_metrics"])
        derivative_stage_values = [
            value
            for key, value in stages.items()
            if key.startswith("derivative")
            or key
            in {
                "build_derivative_stencils",
                "validate_derivative_stencils",
                "run_derivative_siesta_reference",
                "predict_derivative_graph2mat",
                "predict_derivative_deeph",
            }
        ]
        self.assertFalse(any(derivative_stage_values))

    def test_modular_hamiltonian_only_config_disables_derivative_stages(self):
        workflow = _normalized_modular_workflow_payload({"workflow_mode": "hamiltonian_only"})
        stages = workflow["stages"]

        self.assertTrue(stages["hamiltonian_metrics"])
        self.assertFalse(stages["build_derivative_stencils"])
        self.assertFalse(stages["derivative_metrics_graph2mat"])
        self.assertFalse(stages["derivative_metrics_deeph"])
        self.assertFalse(stages["derivative_gate_check"])
        self.assertFalse(stages["derivative_plots"])

    def test_derivative_preflight_bypasses_h_only_workflow(self):
        runner = Graph2MatDeepHBenchmarkRunner()
        workflow = _normalized_modular_workflow_payload({"workflow_mode": "hamiltonian_only"})

        runner._preflight_derivative_workflow(
            {"modular_workflow": workflow},
            stages=workflow["stages"],
            config=workflow["derivative"],
            common_root=Path("/does/not/exist"),
            run_root=None,
            graph2mat_context=None,
            deeph_context=None,
        )

    def test_modular_derivative_metrics_only_does_not_require_training_config(self):
        workflow = _normalized_modular_workflow_payload(
            {
                "workflow_mode": "derivative_metrics_only",
                "derivative": {
                    "enabled": True,
                    "result_dir": "/tmp/existing_derivative_inputs",
                    "method": "central",
                },
            }
        )
        stages = workflow["stages"]

        self.assertFalse(stages["train_graph2mat"])
        self.assertFalse(stages["train_deeph"])
        self.assertFalse(stages["hamiltonian_metrics"])
        self.assertTrue(stages["derivative_metrics_graph2mat"])
        self.assertTrue(stages["derivative_metrics_deeph"])
        self.assertEqual(workflow["derivative"]["method"], "central")

    def test_derivative_preflight_full_build_does_not_require_preexisting_result_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_dataset"
            source.mkdir()
            (root / "existing_graph2mat_predictions").mkdir()
            (root / "existing_deeph_predictions").mkdir()
            payload = {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": str(source),
                    "output_root": str(root / "new_derivative_root"),
                    "method": "central",
                    "delta_ang": 0.01,
                    "atoms": ["0"],
                    "axes": ["x"],
                    "graph2mat_existing_prediction_root": str(root / "existing_graph2mat_predictions"),
                    "deeph_existing_prediction_root": str(root / "existing_deeph_predictions"),
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            runner = Graph2MatDeepHBenchmarkRunner()

            runner._preflight_derivative_workflow(
                payload,
                stages=payload["modular_workflow"]["stages"],
                config=payload["modular_workflow"]["derivative"],
                common_root=root / "new_derivative_root",
                run_root=None,
                graph2mat_context=None,
                deeph_context=None,
            )

    def test_derivative_preflight_existing_reference_root_does_not_require_siesta_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stencil_root = root / "derivative_inputs"
            reference_root = root / "existing_references"
            stencil_root.mkdir()
            reference_root.mkdir()
            payload = {
                "workflow_mode": "derivative_reference_only",
                "derivative": {
                    "enabled": True,
                    "result_dir": str(stencil_root),
                    "method": "central",
                    "existing_reference_root": str(reference_root),
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            runner = Graph2MatDeepHBenchmarkRunner()

            runner._preflight_derivative_workflow(
                payload,
                stages=payload["modular_workflow"]["stages"],
                config=payload["modular_workflow"]["derivative"],
                common_root=stencil_root,
                run_root=None,
                graph2mat_context=None,
                deeph_context=None,
            )

    def test_derivative_siesta_reference_uses_default_or_explicit_command(self):
        for configured_command, expected_command in ((None, "siesta"), ("custom-siesta --flag", "custom-siesta --flag")):
            with self.subTest(configured_command=configured_command):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    stencil_root = root / "derivative_inputs"
                    stencil_root.mkdir()
                    derivative = {
                        "enabled": True,
                        "result_dir": str(stencil_root),
                        "method": "central",
                        "skip_if_exists": False,
                    }
                    if configured_command is not None:
                        derivative["siesta_command"] = configured_command
                    payload = {
                        "workflow_mode": "derivative_reference_only",
                        "derivative": derivative,
                        "stages": {
                            "validate_derivative_stencils": False,
                            "run_derivative_siesta_reference": True,
                        },
                    }
                    payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
                    runner = Graph2MatDeepHBenchmarkRunner()
                    commands = []

                    def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                        commands.append((label, list(command)))
                        reference_root = Path(command[command.index("--output-reference-root") + 1])
                        reference_root.mkdir(parents=True, exist_ok=True)
                        (reference_root / "derivative_siesta_reference_manifest.json").write_text(
                            json.dumps({"samples_failed": 0, "samples_ok": 1}) + "\n",
                            encoding="utf-8",
                        )
                        return {"label": label, "returncode": 0}

                    runner._run_command = fake_run_command  # type: ignore[method-assign]

                    runner._run_modular_derivative_workflow(payload)

                    self.assertEqual([label for label, _command in commands], ["Derivative SIESTA reference Hamiltonians"])
                    command = commands[0][1]
                    self.assertEqual(command[command.index("--siesta-command") + 1], expected_command)

    def test_derivative_preflight_metrics_only_requires_existing_artifact_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_root = Path(tmp) / "missing_derivative_inputs"
            runner = Graph2MatDeepHBenchmarkRunner()
            payload = {
                "workflow_mode": "derivative_metrics_only",
                "derivative": {
                    "enabled": True,
                    "result_dir": str(missing_root),
                    "method": "central",
                },
                "stages": {
                    "derivative_gate_check": False,
                    "derivative_plots": False,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            with self.assertRaisesRegex(RuntimeError, "derivative preflight.*derivative.result_dir"):
                runner._run_modular_derivative_workflow(payload)

    def test_derivative_preflight_deeph_prediction_requires_command_without_existing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stencil_root = root / "derivatives"
            model_dir = root / "deeph_model"
            stencil_root.mkdir()
            model_dir.mkdir()
            runner = Graph2MatDeepHBenchmarkRunner()
            payload = {
                "workflow_mode": "derivative_predictions_only",
                "derivative": {
                    "enabled": True,
                    "result_dir": str(stencil_root),
                    "method": "central",
                    "deeph_model_dir": str(model_dir),
                },
                "stages": {
                    "validate_derivative_stencils": False,
                    "predict_derivative_graph2mat": False,
                    "predict_derivative_deeph": True,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            with self.assertRaisesRegex(RuntimeError, "derivative.deeph_command"):
                runner._run_modular_derivative_workflow(payload)

    def test_modular_derivatives_alias_is_normalized_to_derivative_key(self):
        workflow = _normalized_modular_workflow_payload(
            {
                "workflow_mode": "derivative_metrics_only",
                "derivatives": {
                    "enabled": True,
                    "result_dir": "/tmp/existing_derivative_inputs",
                    "method": "central",
                },
            }
        )

        self.assertIn("derivative", workflow)
        self.assertNotIn("derivatives", workflow)
        self.assertEqual(workflow["derivative"]["result_dir"], "/tmp/existing_derivative_inputs")
        self.assertTrue(workflow["stages"]["derivative_metrics_graph2mat"])

    def test_modular_conflicting_derivative_aliases_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "derivative and derivatives configs conflict"):
            _normalized_modular_workflow_payload(
                {
                    "workflow_mode": "derivative_metrics_only",
                    "derivative": {
                        "enabled": True,
                        "result_dir": "/tmp/one",
                        "method": "central",
                    },
                    "derivatives": {
                        "enabled": True,
                        "result_dir": "/tmp/two",
                        "method": "central",
                    },
                }
            )

    def test_modular_delta_ang_accepts_single_float_and_string_forms(self):
        for value in (0.01, "0.01"):
            workflow = _normalized_modular_workflow_payload(
                {
                    "stages": {"build_derivative_stencils": True},
                    "derivative": {
                        "enabled": True,
                        "source_dataset_root": "/tmp/source",
                        "delta_ang": value,
                        "atoms": [0],
                        "axes": ["x"],
                    },
                }
            )

            self.assertEqual(workflow["derivative"]["delta_ang"], 0.01)
            self.assertEqual(workflow["derivative"]["delta_ang_values"], [0.01])

    def test_modular_delta_ang_accepts_comma_separated_sweep(self):
        workflow = _normalized_modular_workflow_payload(
            {
                "stages": {"build_derivative_stencils": True},
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": "/tmp/source",
                    "delta_ang": "0.005,0.01,0.02",
                    "atoms": [0],
                    "axes": ["x"],
                },
            }
        )

        self.assertEqual(workflow["derivative"]["delta_ang_values"], [0.005, 0.01, 0.02])

    def test_modular_h_then_derivative_full_enables_full_derivative_stage_set(self):
        workflow = _normalized_modular_workflow_payload(
            {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": "/tmp/source",
                    "delta_ang": 0.01,
                    "atoms": [0],
                    "axes": ["x"],
                },
            }
        )
        stages = workflow["stages"]

        self.assertTrue(stages["hamiltonian_metrics"])
        for stage in (
            "build_derivative_stencils",
            "validate_derivative_stencils",
            "run_derivative_siesta_reference",
            "predict_derivative_graph2mat",
            "predict_derivative_deeph",
            "derivative_metrics_graph2mat",
            "derivative_metrics_deeph",
            "derivative_gate_check",
            "derivative_plots",
        ):
            self.assertTrue(stages[stage], stage)

    def test_modular_h_then_derivative_postprocess_remains_metrics_only(self):
        workflow = _normalized_modular_workflow_payload(
            {
                "workflow_mode": "h_then_derivative_postprocess",
                "derivative": {
                    "enabled": True,
                    "method": "central",
                },
            }
        )
        stages = workflow["stages"]

        self.assertFalse(stages["build_derivative_stencils"])
        self.assertFalse(stages["validate_derivative_stencils"])
        self.assertFalse(stages["run_derivative_siesta_reference"])
        self.assertFalse(stages["predict_derivative_graph2mat"])
        self.assertFalse(stages["predict_derivative_deeph"])
        self.assertTrue(stages["derivative_metrics_graph2mat"])
        self.assertTrue(stages["derivative_metrics_deeph"])
        self.assertTrue(stages["derivative_gate_check"])
        self.assertTrue(stages["derivative_plots"])

    def test_modular_invalid_derivative_config_names_missing_field_and_stage(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "build_derivative_stencils.*derivative.source_dataset_root",
        ):
            _normalized_modular_workflow_payload(
                {
                    "stages": {"build_derivative_stencils": True},
                    "derivative": {
                        "enabled": True,
                        "delta_ang": 0.01,
                        "atoms": [0],
                        "axes": ["x"],
                    },
                }
            )

    def test_modular_invalid_derivative_method_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "derivative.method"):
            _normalized_modular_workflow_payload(
                {
                    "workflow_mode": "derivative_metrics_only",
                    "derivative": {
                        "enabled": True,
                        "result_dir": "/tmp/existing_derivative_inputs",
                        "method": "md_snapshots",
                    },
                }
            )

    def test_modular_derivative_predictions_only_uses_separate_model_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "derivatives"
            source = Path(tmp) / "source"
            root.mkdir()
            source.mkdir()
            (Path(tmp) / "g2m_existing").mkdir()
            (Path(tmp) / "deeph_existing").mkdir()
            runner = Graph2MatDeepHBenchmarkRunner()
            commands = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                commands.append((label, list(command)))
                if "geometry validation" in label:
                    root.mkdir(parents=True, exist_ok=True)
                    (root / "derivative_geometry_validation.json").write_text(
                        json.dumps({"errors": 0}) + "\n",
                        encoding="utf-8",
                    )
                elif "graph2mat" in label:
                    output_root = Path(command[command.index("--output-root") + 1])
                    output_root.mkdir(parents=True, exist_ok=True)
                    (output_root / "derivative_graph2mat_prediction_manifest.json").write_text(
                        json.dumps({"samples_failed": 0}) + "\n",
                        encoding="utf-8",
                    )
                elif "deeph" in label:
                    output_root = Path(command[command.index("--output-root") + 1])
                    output_root.mkdir(parents=True, exist_ok=True)
                    (output_root / "derivative_deeph_prediction_manifest.json").write_text(
                        json.dumps({"samples_failed": 0}) + "\n",
                        encoding="utf-8",
                    )
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = {
                "workflow_mode": "derivative_predictions_only",
                "derivative": {
                    "enabled": True,
                    "result_dir": str(root),
                    "source_dataset_root": str(source),
                    "method": "central",
                    "delta_ang": 0.01,
                    "atoms": ["0"],
                    "axes": ["x"],
                    "graph2mat_existing_prediction_root": str(Path(tmp) / "g2m_existing"),
                    "deeph_existing_prediction_root": str(Path(tmp) / "deeph_existing"),
                    "max_samples": 2,
                    "deeph_shell": True,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            summary = runner._run_modular_derivative_workflow(payload)

            labels = [label for label, _command in commands]
            self.assertEqual(
                labels,
                [
                    "Derivative stencil geometry validation",
                    "Derivative graph2mat Hamiltonian predictions",
                    "Derivative deeph Hamiltonian predictions",
                ],
            )
            graph2mat_command = commands[1][1]
            deeph_command = commands[2][1]
            self.assertIn("graph2mat_derivative_result", graph2mat_command[graph2mat_command.index("--stencil-root") + 1])
            self.assertIn("deeph_derivative_result", deeph_command[deeph_command.index("--stencil-root") + 1])
            self.assertEqual(graph2mat_command[graph2mat_command.index("--max-samples") + 1], "2")
            self.assertIn("--python-executable", graph2mat_command)
            self.assertEqual(deeph_command[deeph_command.index("--max-samples") + 1], "2")
            self.assertIn("--deeph-shell", deeph_command)
            self.assertEqual(summary["stages"]["predict_derivative_graph2mat"]["status"], "completed")
            self.assertEqual(summary["stages"]["predict_derivative_deeph"]["status"], "completed")

    def test_modular_derivative_workflow_smoke_runs_ordered_fake_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_dataset"
            source_sample = source / "MD_steps" / "base_0"
            source_sample.mkdir(parents=True)
            (source_sample / "RUN.fdf").write_text("SystemLabel smoke\n", encoding="utf-8")
            graph_existing = root / "existing_graph2mat_predictions"
            deeph_existing = root / "existing_deeph_predictions"
            graph_existing.mkdir()
            deeph_existing.mkdir()
            derivative_root = root / "derivative_smoke"
            runner = Graph2MatDeepHBenchmarkRunner()
            commands = []

            def write_metric_outputs(output_dir: Path, model: str) -> None:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "manifest.json").write_text(
                    json.dumps({"status": "ok", "source_model": model}) + "\n",
                    encoding="utf-8",
                )
                _write_csv(
                    output_dir / "derivative_matrix_metrics.csv",
                    [
                        {
                            "sample": f"{model}_sample",
                            "base_sample_id": "base_0",
                            "atom_index_zero_based": "0",
                            "axis": "x",
                            "delta_ang": "0.01",
                            "finite_difference_method": "central",
                            "dh_mae_union_eV_per_Ang": "0.1",
                            "comparison_status": "diagnostic_only",
                        }
                    ],
                )

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                commands.append((label, list(command)))
                if label == "Derivative stencil builder":
                    output_root = Path(command[command.index("--output-stencil-root") + 1])
                    (output_root / "structures" / "base_0__atom0_x_plus").mkdir(parents=True)
                    (output_root / "derivative_stencil_manifest.json").write_text(
                        json.dumps({"sample_count": 3, "stencil_count": 1}) + "\n",
                        encoding="utf-8",
                    )
                elif label == "Derivative stencil geometry validation":
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "derivative_geometry_validation.json").write_text(
                        json.dumps({"errors": 0, "warnings": 0}) + "\n",
                        encoding="utf-8",
                    )
                elif label == "Derivative SIESTA reference Hamiltonians":
                    reference_root = Path(command[command.index("--output-reference-root") + 1])
                    reference_root.mkdir(parents=True, exist_ok=True)
                    (reference_root / "derivative_siesta_reference_manifest.json").write_text(
                        json.dumps({"samples_failed": 0, "samples_ok": 2}) + "\n",
                        encoding="utf-8",
                    )
                elif label in {
                    "Derivative graph2mat Hamiltonian predictions",
                    "Derivative deeph Hamiltonian predictions",
                }:
                    model = command[command.index("--model") + 1]
                    output_root = Path(command[command.index("--output-root") + 1])
                    output_root.mkdir(parents=True, exist_ok=True)
                    (output_root / f"derivative_{model}_prediction_manifest.json").write_text(
                        json.dumps({"samples_failed": 0, "samples_ok": 2}) + "\n",
                        encoding="utf-8",
                    )
                elif label in {
                    "Derivative graph2mat finite-difference metrics",
                    "Derivative deeph finite-difference metrics",
                }:
                    model = "graph2mat" if "graph2mat" in label else "deeph"
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    write_metric_outputs(output_dir, model)
                else:
                    raise AssertionError(f"Unexpected command label: {label}")
                return {"label": label, "returncode": 0}

            def fake_plot_outputs(*, derivative_roots, graph2mat_root, deeph_root, output_dir):
                output_dir.mkdir(parents=True, exist_ok=True)
                payload_path = output_dir / "derivative_plot_payload.json"
                manifest_path = output_dir / "derivative_plot_manifest.json"
                payload_path.write_text(json.dumps({"available": True}) + "\n", encoding="utf-8")
                manifest_path.write_text(json.dumps({"available": True}) + "\n", encoding="utf-8")
                return {
                    "payload": {"available": True},
                    "manifest": {"available": True},
                    "payload_path": payload_path,
                    "manifest_path": manifest_path,
                    "graph2mat_root": str(graph2mat_root),
                    "deeph_root": str(deeph_root),
                    "derivative_roots": [str(path) for path in derivative_roots],
                }

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": str(source),
                    "output_root": str(derivative_root),
                    "method": "central",
                    "delta_ang": 0.01,
                    "base_split": "test",
                    "atoms": ["0"],
                    "axes": ["x"],
                    "siesta_command": "siesta",
                    "graph2mat_existing_prediction_root": str(graph_existing),
                    "deeph_existing_prediction_root": str(deeph_existing),
                    "skip_if_exists": False,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            with (
                mock.patch("g2m_deeph_runner.write_derivative_plot_outputs", side_effect=fake_plot_outputs),
                mock.patch(
                    "g2m_deeph_runner.build_derivative_gate_report",
                    return_value={
                        "schema_version": "graph2mat_deeph_derivative_gate_report_v1",
                        "scientific_status": "internal_diagnostic",
                        "blockers": [],
                        "warnings": [],
                    },
                ),
            ):
                summary = runner._run_modular_derivative_workflow(payload)

            self.assertEqual(
                [label for label, _command in commands],
                [
                    "Derivative stencil builder",
                    "Derivative stencil geometry validation",
                    "Derivative SIESTA reference Hamiltonians",
                    "Derivative graph2mat Hamiltonian predictions",
                    "Derivative deeph Hamiltonian predictions",
                    "Derivative graph2mat finite-difference metrics",
                    "Derivative deeph finite-difference metrics",
                ],
            )
            graph_command = commands[3][1]
            deeph_command = commands[4][1]
            graph_root = Path(graph_command[graph_command.index("--stencil-root") + 1])
            deeph_root = Path(deeph_command[deeph_command.index("--stencil-root") + 1])
            self.assertIn("graph2mat_derivative_result", str(graph_root))
            self.assertIn("deeph_derivative_result", str(deeph_root))
            self.assertNotEqual(graph_root, deeph_root)
            graph_metric_command = commands[5][1]
            deeph_metric_command = commands[6][1]
            self.assertEqual(Path(graph_metric_command[2]), graph_root)
            self.assertEqual(Path(deeph_metric_command[2]), deeph_root)
            manifest_path = derivative_root / "derivative_workflow_manifest.json"
            self.assertTrue(manifest_path.exists())
            written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(written_manifest["stages"]["derivative_model_comparison"]["summary"]["claim_status"], "diagnostic_only")
            self.assertFalse(written_manifest["stages"]["derivative_model_comparison"]["summary"]["winner_claim_allowed"])
            self.assertIsNone(written_manifest["stages"]["derivative_model_comparison"]["summary"]["winner"])
            self.assertEqual(summary["stages"]["derivative_model_comparison"]["summary"]["paired_count"], 1)

            h_only = {"workflow_mode": "hamiltonian_only"}
            h_only["modular_workflow"] = _normalized_modular_workflow_payload(h_only)
            command_count = len(commands)
            h_summary = runner._run_modular_derivative_workflow(h_only, run_root=root / "h_only")
            self.assertEqual(len(commands), command_count)
            self.assertEqual(h_summary["stages"], {})

    def test_h_then_derivative_full_infers_prediction_model_artifacts_from_h_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "derivatives").mkdir()
            graph_context, deeph_context = self._benchmark_contexts(root)
            checkpoint = graph_context.training_dir / "lightning_logs" / "version_0" / "checkpoints" / "best.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("checkpoint\n", encoding="utf-8")
            (graph_context.training_dir / "checkpoint_manifest.json").write_text(
                json.dumps({"checkpoint_path": str(checkpoint)}) + "\n",
                encoding="utf-8",
            )
            deeph_context.save_dir.mkdir(parents=True)
            runner = Graph2MatDeepHBenchmarkRunner()
            commands = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                commands.append((label, list(command)))
                output_root = Path(command[command.index("--output-root") + 1])
                output_root.mkdir(parents=True, exist_ok=True)
                model = command[command.index("--model") + 1]
                (output_root / f"derivative_{model}_prediction_manifest.json").write_text(
                    json.dumps({"samples_failed": 0}) + "\n",
                    encoding="utf-8",
                )
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = self._h_then_derivative_prediction_payload(root / "derivatives")

            summary = runner._run_modular_derivative_workflow(
                payload,
                run_root=graph_context.run_root,
                graph2mat_context=graph_context,
                deeph_context=deeph_context,
            )

            graph2mat_command = commands[0][1]
            deeph_command = commands[1][1]
            self.assertEqual(graph2mat_command[graph2mat_command.index("--checkpoint") + 1], str(checkpoint))
            self.assertEqual(deeph_command[deeph_command.index("--model-dir") + 1], str(deeph_context.save_dir))
            self.assertEqual(
                summary["stages"]["predict_derivative_graph2mat"]["model_artifact"]["source"],
                "inferred_h_workflow_checkpoint",
            )
            self.assertEqual(
                summary["stages"]["predict_derivative_deeph"]["model_artifact"]["source"],
                "inferred_h_workflow_model_dir",
            )

    def test_h_then_derivative_full_explicit_prediction_artifacts_override_inferred_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "derivatives").mkdir()
            graph_context, deeph_context = self._benchmark_contexts(root)
            inferred_checkpoint = graph_context.training_dir / "lightning_logs" / "version_0" / "checkpoints" / "best.ckpt"
            inferred_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            inferred_checkpoint.write_text("checkpoint\n", encoding="utf-8")
            (graph_context.training_dir / "checkpoint_manifest.json").write_text(
                json.dumps({"checkpoint_path": str(inferred_checkpoint)}) + "\n",
                encoding="utf-8",
            )
            deeph_context.save_dir.mkdir(parents=True)
            explicit_checkpoint = root / "explicit" / "graph2mat.ckpt"
            explicit_checkpoint.parent.mkdir(parents=True)
            explicit_checkpoint.write_text("checkpoint\n", encoding="utf-8")
            explicit_model_dir = root / "explicit" / "deeph_model"
            explicit_model_dir.mkdir(parents=True)
            runner = Graph2MatDeepHBenchmarkRunner()
            commands = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                commands.append((label, list(command)))
                output_root = Path(command[command.index("--output-root") + 1])
                output_root.mkdir(parents=True, exist_ok=True)
                model = command[command.index("--model") + 1]
                (output_root / f"derivative_{model}_prediction_manifest.json").write_text(
                    json.dumps({"samples_failed": 0}) + "\n",
                    encoding="utf-8",
                )
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = self._h_then_derivative_prediction_payload(
                root / "derivatives",
                derivative={
                    "graph2mat_checkpoint": str(explicit_checkpoint),
                    "deeph_model_dir": str(explicit_model_dir),
                },
            )

            summary = runner._run_modular_derivative_workflow(
                payload,
                run_root=graph_context.run_root,
                graph2mat_context=graph_context,
                deeph_context=deeph_context,
            )

            graph2mat_command = commands[0][1]
            deeph_command = commands[1][1]
            self.assertEqual(graph2mat_command[graph2mat_command.index("--checkpoint") + 1], str(explicit_checkpoint))
            self.assertEqual(deeph_command[deeph_command.index("--model-dir") + 1], str(explicit_model_dir))
            self.assertEqual(
                summary["stages"]["predict_derivative_graph2mat"]["model_artifact"]["source"],
                "configured_graph2mat_checkpoint",
            )
            self.assertEqual(
                summary["stages"]["predict_derivative_deeph"]["model_artifact"]["source"],
                "configured_deeph_model_dir",
            )

    def test_h_then_derivative_full_existing_prediction_roots_do_not_require_model_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "derivatives").mkdir()
            (root / "existing_graph2mat_predictions").mkdir()
            (root / "existing_deeph_predictions").mkdir()
            graph_context, deeph_context = self._benchmark_contexts(root)
            runner = Graph2MatDeepHBenchmarkRunner()
            commands = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                commands.append((label, list(command)))
                output_root = Path(command[command.index("--output-root") + 1])
                output_root.mkdir(parents=True, exist_ok=True)
                model = command[command.index("--model") + 1]
                (output_root / f"derivative_{model}_prediction_manifest.json").write_text(
                    json.dumps({"samples_failed": 0}) + "\n",
                    encoding="utf-8",
                )
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = self._h_then_derivative_prediction_payload(
                root / "derivatives",
                derivative={
                    "graph2mat_existing_prediction_root": str(root / "existing_graph2mat_predictions"),
                    "deeph_existing_prediction_root": str(root / "existing_deeph_predictions"),
                },
            )

            runner._run_modular_derivative_workflow(
                payload,
                run_root=graph_context.run_root,
                graph2mat_context=graph_context,
                deeph_context=deeph_context,
            )

            for _label, command in commands:
                self.assertIn("--existing-prediction-root", command)
                self.assertNotIn("--checkpoint", command)
                self.assertNotIn("--model-dir", command)

    def test_h_then_derivative_full_missing_inferred_graph2mat_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "derivatives").mkdir()
            graph_context, deeph_context = self._benchmark_contexts(root)
            runner = Graph2MatDeepHBenchmarkRunner()
            payload = self._h_then_derivative_prediction_payload(
                root / "derivatives",
                stages={"predict_derivative_deeph": False},
            )

            with self.assertRaises(RuntimeError) as raised:
                runner._run_modular_derivative_workflow(
                    payload,
                    run_root=graph_context.run_root,
                    graph2mat_context=graph_context,
                    deeph_context=deeph_context,
                )

            self.assertIn("derivative.graph2mat_checkpoint", str(raised.exception))
            self.assertIn("derivative.graph2mat_existing_prediction_root", str(raised.exception))

    def test_h_then_derivative_full_missing_inferred_deeph_model_dir_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "derivatives").mkdir()
            graph_context, deeph_context = self._benchmark_contexts(root)
            runner = Graph2MatDeepHBenchmarkRunner()
            payload = self._h_then_derivative_prediction_payload(
                root / "derivatives",
                stages={"predict_derivative_graph2mat": False},
            )

            with self.assertRaises(RuntimeError) as raised:
                runner._run_modular_derivative_workflow(
                    payload,
                    run_root=graph_context.run_root,
                    graph2mat_context=graph_context,
                    deeph_context=deeph_context,
                )

            self.assertIn("derivative.deeph_model_dir", str(raised.exception))
            self.assertIn("derivative.deeph_existing_prediction_root", str(raised.exception))

    def test_modular_stencil_builder_receives_delta_sweep_and_include_base_for_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "derivatives"
            source = Path(tmp) / "source"
            source.mkdir()
            runner = Graph2MatDeepHBenchmarkRunner()
            commands = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                commands.append((label, list(command)))
                if "stencil builder" in label:
                    root.mkdir(parents=True, exist_ok=True)
                    (root / "derivative_stencil_manifest.json").write_text(
                        json.dumps({"sample_count": 3, "stencil_count": 3}) + "\n",
                        encoding="utf-8",
                    )
                elif "geometry validation" in label:
                    (root / "derivative_geometry_validation.json").write_text(
                        json.dumps({"errors": 0}) + "\n",
                        encoding="utf-8",
                    )
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = {
                "workflow_mode": "derivative_stencils_only",
                "derivative": {
                    "enabled": True,
                    "result_dir": str(root),
                    "source_dataset_root": str(source),
                    "method": "central",
                    "delta_ang": [0.005, 0.01, 0.02],
                    "atoms": ["0"],
                    "axes": ["x"],
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            summary = runner._run_modular_derivative_workflow(payload)

            builder_command = commands[0][1]
            delta_start = builder_command.index("--delta-ang") + 1
            delta_end = builder_command.index("--split")
            self.assertEqual(builder_command[delta_start:delta_end], ["0.005", "0.01", "0.02"])
            self.assertIn("--include-base", builder_command)
            self.assertEqual(summary["stages"]["build_derivative_stencils"]["status"], "completed")
            self.assertEqual(summary["stages"]["validate_derivative_stencils"]["status"], "completed")

    def test_modular_reference_stage_uses_max_samples_and_explicit_siesta_shell_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "derivatives"
            root.mkdir()
            runner = Graph2MatDeepHBenchmarkRunner()
            commands = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                commands.append((label, list(command)))
                if "geometry validation" in label:
                    (root / "derivative_geometry_validation.json").write_text(
                        json.dumps({"errors": 0}) + "\n",
                        encoding="utf-8",
                    )
                elif "SIESTA reference" in label:
                    reference_root = Path(command[command.index("--output-reference-root") + 1])
                    reference_root.mkdir(parents=True, exist_ok=True)
                    (reference_root / "derivative_siesta_reference_manifest.json").write_text(
                        json.dumps({"samples_failed": 0}) + "\n",
                        encoding="utf-8",
                    )
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = {
                "workflow_mode": "derivative_reference_only",
                "derivative": {
                    "enabled": True,
                    "result_dir": str(root),
                    "method": "central",
                    "siesta_command": "siesta",
                    "max_samples": 2,
                    "siesta_shell": True,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            summary = runner._run_modular_derivative_workflow(payload)

            reference_command = commands[1][1]
            self.assertEqual(reference_command[reference_command.index("--max-samples") + 1], "2")
            self.assertIn("--siesta-shell", reference_command)
            self.assertEqual(summary["stages"]["run_derivative_siesta_reference"]["status"], "completed")

    def test_modular_derivative_metrics_only_start_skips_dataset_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "existing_derivative_inputs"
            root.mkdir()
            runner = Graph2MatDeepHBenchmarkRunner()

            def fail_dataset_validation(_payload):
                raise AssertionError("derivative-only workflow should not validate the H benchmark dataset")

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "manifest.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
                return {"label": label, "returncode": 0}

            runner.validate_dataset_payload = fail_dataset_validation  # type: ignore[method-assign]
            runner._run_command = fake_run_command  # type: ignore[method-assign]
            status = runner.start(
                {
                    "workflow_mode": "derivative_metrics_only",
                    "run_id": "derivative_metrics_only_test",
                    "derivative": {
                        "enabled": True,
                        "result_dir": str(root),
                        "method": "central",
                    },
                    "stages": {
                        "derivative_gate_check": False,
                        "derivative_plots": False,
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)
            final_status = runner.status()

            self.assertFalse(final_status["running"])
            self.assertEqual(final_status["returncode"], 0)
            self.assertIn("derivative_only_workflow", final_status["dataset_validation"]["warnings"][0])

    def test_h_then_derivative_postprocess_uses_h_metric_inputs_when_result_dir_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "h_run"
            (run_root / "common_metrics" / "graph2mat_eval").mkdir(parents=True)
            (run_root / "common_metrics" / "deeph_eval").mkdir(parents=True)
            runner = Graph2MatDeepHBenchmarkRunner()
            result_dirs = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **_kwargs):
                result_dirs.append(Path(command[2]))
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "manifest.json").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            payload = {
                "workflow_mode": "h_then_derivative_postprocess",
                "derivative": {
                    "enabled": True,
                    "method": "central",
                },
                "stages": {
                    "derivative_gate_check": False,
                    "derivative_plots": False,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            summary = runner._run_modular_derivative_workflow(payload, run_root=run_root)

            self.assertEqual(
                result_dirs,
                [
                    run_root / "common_metrics" / "graph2mat_eval",
                    run_root / "common_metrics" / "deeph_eval",
                ],
            )
            self.assertEqual(summary["stages"]["derivative_metrics_graph2mat"]["status"], "completed")
            self.assertEqual(summary["stages"]["derivative_metrics_deeph"]["status"], "completed")

    def test_derivative_model_comparison_pairs_matching_stencils_without_winner_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_root = root / "graph2mat" / "derivative_metrics"
            deeph_root = root / "deeph" / "derivative_metrics"
            for metric_root, sample, mae in (
                (graph_root, "g2m_dH", "0.20"),
                (deeph_root, "deeph_dH", "0.10"),
            ):
                _write_csv(
                    metric_root / "derivative_matrix_metrics.csv",
                    [
                        {
                            "sample": sample,
                            "base_sample_id": "base_0",
                            "atom_index_zero_based": "0",
                            "axis": "x",
                            "delta_ang": "0.01",
                            "finite_difference_method": "central",
                            "dh_mae_union_eV_per_Ang": mae,
                            "dh_rmse_union_eV_per_Ang": mae,
                            "comparison_status": "diagnostic_only",
                        }
                    ],
                )

            summary = build_derivative_model_comparison_summary(
                graph2mat_root=graph_root,
                deeph_root=deeph_root,
                output_dir=root / "comparison",
                gate_report={"scientific_status": "internal_diagnostic"},
            )

            self.assertEqual(summary["paired_count"], 1)
            self.assertEqual(summary["claim_status"], "diagnostic_only")
            self.assertFalse(summary["winner_claim_allowed"])
            self.assertIsNone(summary["winner"])
            self.assertEqual(summary["block_metrics"]["status"], "block_metrics_unavailable")
            with (root / "comparison" / "derivative_model_paired_comparison.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["base_sample_id"], "base_0")
            self.assertAlmostEqual(float(rows[0]["delta_graph2mat_minus_deeph_dh_mae_union_eV_per_Ang"]), 0.10)

    def test_derivative_model_comparison_handles_missing_model_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_root = root / "graph2mat" / "derivative_metrics"
            _write_csv(
                graph_root / "derivative_matrix_metrics.csv",
                [
                    {
                        "sample": "g2m_dH",
                        "base_sample_id": "base_0",
                        "atom_index_zero_based": "0",
                        "axis": "x",
                        "delta_ang": "0.01",
                        "finite_difference_method": "central",
                        "dh_mae_union_eV_per_Ang": "0.20",
                    }
                ],
            )

            summary = build_derivative_model_comparison_summary(
                graph2mat_root=graph_root,
                deeph_root=None,
                output_dir=root / "comparison",
                gate_report={"scientific_status": "blocked"},
            )

            self.assertEqual(summary["paired_count"], 0)
            self.assertEqual(summary["missing_deeph_count"], 1)
            self.assertEqual(summary["claim_status"], "blocked")
            self.assertIsNone(summary["winner"])
            self.assertTrue((root / "comparison" / "derivative_model_comparison_summary.json").exists())

    def test_deeph_metric_command_can_target_validation_split(self):
        command = _deeph_metric_command_args(
            python_executable="python3",
            graph2mat_result_dir=Path("g2m"),
            processed_dir=Path("processed"),
            predictions_dir=Path("predictions"),
            output_dir=Path("out"),
            metric_fail_policy=METRIC_FAIL_POLICY_FAIL_CLOSED,
            split="validation",
        )

        self.assertEqual(command[command.index("--split") + 1], "validation")

    def test_search_composite_evaluation_uses_validation_split(self):
        payload = {
            "benchmark_mode": "final_publication",
            "protocol_stage": "search",
            "protocol": {
                "selection": {"metric": "val_spectral_composite", "mode": "min"},
                "top_k_selection": {"k_per_model": 1},
            },
        }

        self.assertEqual(_metric_evaluation_split(payload), "validation")

    def test_search_metric_evaluation_rejects_locked_test_split(self):
        payload = {
            "benchmark_mode": "final_publication",
            "protocol_stage": "search",
            "metric_evaluation_split": "test",
            "protocol": {
                "search_evaluation": {"run_validation_metrics": True},
                "selection": {"metric": "val_spectral_composite", "mode": "min"},
                "top_k_selection": {"k_per_model": 1},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "locked test split"):
            _metric_evaluation_split(payload)

    def test_robust_validation_composite_evaluation_uses_validation_split(self):
        payload = {
            "benchmark_mode": "final_publication",
            "protocol_stage": "robust_validation",
            "protocol": {
                "search_evaluation": {"run_validation_metrics": True},
                "selection": {"metric": "val_spectral_composite", "mode": "min"},
                "top_k_selection": {"k_per_model": 1},
            },
        }

        self.assertEqual(_metric_evaluation_split(payload), "validation")

    def test_robust_validation_metric_evaluation_rejects_locked_test_split(self):
        payload = {
            "benchmark_mode": "final_publication",
            "protocol_stage": "robust_validation",
            "metric_evaluation_split": "test",
            "protocol": {
                "search_evaluation": {"run_validation_metrics": True},
                "selection": {"metric": "val_spectral_composite", "mode": "min"},
                "top_k_selection": {"k_per_model": 1},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "locked test split"):
            _metric_evaluation_split(payload)

    def test_extract_validation_metrics_from_kpoint_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            (metrics / "kpoint_spectral_metrics.csv").write_text(
                "sample,low_energy_rmse_eV,fermi_window_rmse_eV,frontier_window_rmse_eV,global_rmse_eV\n"
                "a,0.2,0.3,0.4,0.5\n"
                "b,0.4,0.5,0.6,0.7\n",
                encoding="utf-8",
            )
            (metrics / "kpoint_dos_metrics.csv").write_text(
                "sample,dos_wasserstein_eV,dos_mae_500_fermi_window\n"
                "a,0.02,0.03\n"
                "b,0.04,0.05\n",
                encoding="utf-8",
            )
            (metrics / "kpoint_matrix_metrics.csv").write_text(
                "sample,row_type,h_mae_eV\n"
                "a,weighted_sample,0.1\n"
                "b,weighted_sample,0.3\n",
                encoding="utf-8",
            )

            extracted = _extract_validation_metrics(metrics)

        self.assertAlmostEqual(extracted["low_energy_rmse_eV"], 0.3)
        self.assertAlmostEqual(extracted["global_band_rmse"], 0.6)
        self.assertAlmostEqual(extracted["dos_wasserstein"], 0.03)
        self.assertAlmostEqual(extracted["dos_mae_near_fermi"], 0.04)
        self.assertAlmostEqual(extracted["h_mae_eV"], 0.2)

    def test_diagnostic_metric_manifest_disables_robust_winners(self):
        manifest = {
            "status": "valid_reused_joint_dataset",
            "warnings": [],
            "recommendation": {"winner": "deeph", "robust_recommendation": True},
        }

        updated = _force_diagnostic_metric_manifest(
            manifest,
            metric_fail_policy=METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
        )

        self.assertEqual(updated["status"], "diagnostic_only")
        self.assertEqual(updated["comparability_status"], "diagnostic_only")
        self.assertFalse(updated["robust_winner_allowed"])
        self.assertIsNone(updated["recommendation"]["winner"])
        self.assertFalse(updated["recommendation"]["robust_recommendation"])

    def test_deeph_pack_root_env_is_used_for_cli_discovery(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"DEEPH_PACK_ROOT": str(Path(tmp) / "deeph_env"), "PATH": ""}, clear=True):
            repo = Path(os.environ["DEEPH_PACK_ROOT"])
            cli = _write_fake_deeph_cli(repo, "deeph-train")
            runner = Graph2MatDeepHBenchmarkRunner()

            self.assertEqual(runner._deeph_command({}, "deeph-train"), str(cli))
            discovery = runner._deeph_discovery({})
            self.assertEqual(discovery["source"], "env")
            self.assertEqual(discovery["repo_path"], str(repo.resolve()))

    def test_configured_deeph_repo_path_overrides_env(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"DEEPH_PACK_ROOT": str(Path(tmp) / "deeph_env"), "PATH": ""}, clear=True):
            env_repo = Path(os.environ["DEEPH_PACK_ROOT"])
            config_repo = Path(tmp) / "deeph_config"
            _write_fake_deeph_cli(env_repo, "deeph-preprocess")
            cli = _write_fake_deeph_cli(config_repo, "deeph-preprocess")
            runner = Graph2MatDeepHBenchmarkRunner()

            self.assertEqual(
                runner._deeph_command({"deeph": {"repo_path": str(config_repo)}}, "deeph-preprocess"),
                str(cli),
            )
            discovery = runner._deeph_discovery({"deeph": {"repo_path": str(config_repo)}})
            self.assertEqual(discovery["source"], "config")
            self.assertEqual(discovery["repo_path"], str(config_repo.resolve()))

    def test_deeph_cli_path_discovery_works(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"PATH": str(Path(tmp) / "bin")}, clear=True):
            cli = _write_fake_deeph_cli(Path(tmp) / "bin", "deeph-inference")
            runner = Graph2MatDeepHBenchmarkRunner()

            self.assertEqual(runner._deeph_command({}, "deeph-inference"), str(cli))
            discovery = runner._deeph_discovery({})
            self.assertEqual(discovery["source"], "PATH")

    def test_sibling_deeph_pack_repo_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
            repo_root = Path(tmp) / "MD_vs_AtomicDisplacement"
            repo_root.mkdir()
            sibling_repo = Path(tmp) / "DeepH-pack"
            cli = _write_fake_deeph_cli(sibling_repo, "deeph-preprocess")
            runner = Graph2MatDeepHBenchmarkRunner()

            with mock.patch("g2m_deeph_runner.REPO_ROOT", repo_root):
                self.assertEqual(runner._deeph_command({}, "deeph-preprocess"), str(cli))
                discovery = runner._deeph_discovery({})

            self.assertEqual(discovery["source"], "sibling_repo")
            self.assertEqual(discovery["repo_path"], str(sibling_repo.resolve()))

    def test_deeph_command_env_prefers_discovered_source_package(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
            repo_root = Path(tmp) / "MD_vs_AtomicDisplacement"
            repo_root.mkdir()
            sibling_repo = Path(tmp) / "DeepH-pack"
            (sibling_repo / "deeph").mkdir(parents=True)
            _write_fake_deeph_cli(sibling_repo, "deeph-preprocess")
            python = sibling_repo / ".venv" / "bin" / "python"
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            runner = Graph2MatDeepHBenchmarkRunner()

            with mock.patch("g2m_deeph_runner.REPO_ROOT", repo_root):
                env = runner._deeph_command_env({})

            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(sibling_repo.resolve()))
            self.assertTrue(env["PATH"].split(os.pathsep)[0].endswith("DeepH-pack/.venv/bin"))

    def test_missing_deeph_cli_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
            repo_root = Path(tmp) / "MD_vs_AtomicDisplacement"
            repo_root.mkdir()
            runner = Graph2MatDeepHBenchmarkRunner()

            with mock.patch("g2m_deeph_runner.REPO_ROOT", repo_root):
                with self.assertRaisesRegex(RuntimeError, "DEEPH_PACK_ROOT"):
                    runner._deeph_command({}, "deeph-train")

    def test_no_user_local_deeph_pack_path_remains_in_g2m_sources(self):
        hardcoded_repo = "/" + "/".join(("home", "christian", "repositorios", "DeepH" + "-pack"))
        hardcoded_venv = "DeepH" + "-pack/.venv"
        for path in (
            SCRIPTS_DIR / "g2m_deeph_runner.py",
            SCRIPTS_DIR / "pipeline_ui.py",
            SCRIPTS_DIR / "deeph_fair_utils.py",
            REPO_ROOT / "Comparison" / "ui" / "index.html",
            REPO_ROOT / "Comparison" / "ui" / "app.js",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(hardcoded_repo, text)
            self.assertNotIn(hardcoded_venv, text)

    def test_production_metric_policy_rejects_returncode_two(self):
        runner = Graph2MatDeepHBenchmarkRunner()

        with self.assertRaisesRegex(RuntimeError, "exit code 2"):
            runner._run_command(
                [sys.executable, "-c", "import sys; sys.exit(2)"],
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                label="unit metric rc2",
                allowed_returncodes=_metric_allowed_returncodes(METRIC_FAIL_POLICY_FAIL_CLOSED),
            )

    def test_run_command_failure_record_classifies_cuda_oom(self):
        runner = Graph2MatDeepHBenchmarkRunner()

        with self.assertRaises(CommandRunError) as ctx:
            runner._run_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('RuntimeError: CUDA out of memory'); sys.exit(1)",
                ],
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                label="unit cuda oom",
            )

        telemetry = ctx.exception.run_record["telemetry"]
        self.assertEqual(telemetry["failure_category"], "cuda_oom_detected")
        self.assertIn("CUDA out of memory", telemetry["failure_evidence_excerpt"])
        self.assertIn("returncode", ctx.exception.run_record)

    def test_diagnostic_metric_policy_may_accept_returncode_two(self):
        runner = Graph2MatDeepHBenchmarkRunner()

        result = runner._run_command(
            [sys.executable, "-c", "import sys; sys.exit(2)"],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            label="unit metric diagnostic rc2",
            allowed_returncodes=_metric_allowed_returncodes(METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY),
        )

        self.assertEqual(result["returncode"], 2)
        self.assertEqual(result["telemetry"]["failure_category"], "nonzero_exit")

    def test_run_command_emits_training_progress_heartbeat(self):
        runner = Graph2MatDeepHBenchmarkRunner()

        with mock.patch("g2m_deeph_runner.LOG_HEARTBEAT_SECONDS", 0.05):
            runner._run_command(
                [sys.executable, "-c", "import time; time.sleep(0.18)"],
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                label="unit graph2mat train",
                progress_provider=lambda: "epoch 3/10 | train_epoch 0.1234",
            )

        logs = "".join(runner.logs(since=0, limit=1000)["lines"])
        self.assertIn("[G2M-DEEPH][PROGRESS] unit graph2mat train", logs)
        self.assertIn("epoch 3/10", logs)

    def test_validate_dataset_returns_artifact_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_dataset_provenance(dataset)
            _write_snapshot(dataset, "0", complete=True)

            runner = Graph2MatDeepHBenchmarkRunner()
            payload = runner.validate_dataset_payload({"dataset_root": str(dataset)})

            self.assertTrue(payload["benchmark_ready"])
            self.assertEqual(payload["artifact_summary"]["total_snapshots"], 1)
            self.assertEqual(payload["artifact_summary"]["valid_snapshots"], 1)
            self.assertEqual(payload["artifact_summary"]["missing_required_counts"], {})

    def test_runner_requires_dataset_provenance_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_snapshot(dataset, "0", complete=True)

            runner = Graph2MatDeepHBenchmarkRunner()
            payload = runner.validate_dataset_payload({"dataset_root": str(dataset)})

            self.assertFalse(payload["benchmark_ready"])
            self.assertIn("dataset-level basis provenance", "\n".join(payload["errors"]))
            self.assertIn("dataset-level pseudopotential provenance", "\n".join(payload["errors"]))
            self.assertIn("dataset-level material identity", "\n".join(payload["errors"]))

    def test_validate_dataset_auto_selects_single_valid_child_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "graphene_w90_joint"
            child = parent / "md_sweep_1"
            _write_training_ready_dataset(child)
            (child / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")

            runner = Graph2MatDeepHBenchmarkRunner()
            payload = runner.validate_dataset_payload({"dataset_root": str(parent)})

            self.assertTrue(payload["benchmark_ready"])
            self.assertTrue(payload["auto_selected_child_dataset"])
            self.assertEqual(Path(payload["dataset_root"]), child)
            self.assertEqual(Path(payload["dataset_collection_root"]), parent)
            self.assertEqual(payload["artifact_summary"]["missing_required_counts"], {})

    def test_validate_dataset_uses_persistent_comparison_dataset_default(self):
        runner = Graph2MatDeepHBenchmarkRunner()

        payload = runner.validate_dataset_payload({})

        if payload.get("auto_selected_child_dataset"):
            self.assertEqual(Path(payload["dataset_collection_root"]), DEFAULT_DATASET_ROOT)
            self.assertTrue(Path(payload["dataset_root"]).is_relative_to(DEFAULT_DATASET_ROOT))
        else:
            self.assertEqual(Path(payload["dataset_root"]), DEFAULT_DATASET_ROOT)
            self.assertIn("Comparison/datasets/graphene_w90_joint", payload["dataset_root"])
            self.assertFalse(payload["benchmark_ready"])

    def test_old_graph2mat_only_dataset_is_not_benchmark_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_dataset_provenance(dataset)
            _write_snapshot(dataset, "0", complete=False)

            runner = Graph2MatDeepHBenchmarkRunner()
            payload = runner.validate_dataset_payload({"dataset_root": str(dataset)})

            self.assertFalse(payload["benchmark_ready"])
            self.assertTrue(payload["repair_required"])
            self.assertEqual(payload["artifact_summary"]["missing_required_counts"]["hsx"], 1)
            self.assertEqual(payload["artifact_summary"]["missing_required_counts"]["struct_out"], 1)
            self.assertEqual(payload["artifact_summary"]["missing_required_counts"]["orb_indx"], 1)

    def test_run_rejects_missing_artifacts_without_explicit_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_dataset_provenance(dataset)
            _write_snapshot(dataset, "0", complete=False)

            runner = Graph2MatDeepHBenchmarkRunner()
            with self.assertRaisesRegex(RuntimeError, "hsx=1"):
                runner.start({"dataset_root": str(dataset)})

    def test_run_rejects_empty_default_dataset_with_actionable_message(self):
        runner = Graph2MatDeepHBenchmarkRunner()

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "not benchmark-ready"):
                runner.start({"dataset_root": str(Path(tmp) / "empty_dataset")})

    def test_generate_new_mode_plans_single_md_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()
            status = runner.start(
                {
                    "dataset_root": str(Path(tmp) / "datasets"),
                    "output_root": str(Path(tmp) / "results"),
                    "dataset_mode": "generate_new",
                    "snapshot_count": 6,
                    "dry_run": True,
                    "split_mode": "block",
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)
            results = runner.results()["results"]
            self.assertEqual(results["dataset_sweep"]["total_datasets"], 1)
            row = results["dataset_sweep"]["rows"][0]
            self.assertEqual(row["dataset_size"], 6)
            self.assertEqual(row["status"], "dry_run")

    def test_dataset_sweep_dry_run_plans_multiple_md_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()
            status = runner.start(
                {
                    "dataset_root": str(Path(tmp) / "datasets"),
                    "output_root": str(Path(tmp) / "results"),
                    "dataset_mode": "generate_new",
                    "run_mode": "generate_datasets_only",
                    "dry_run": True,
                    "split_mode": "block",
                    "dataset_sweep": {
                        "enabled": True,
                        "max_datasets": 2,
                        "recipes": [
                            {"recipe_id": "md_a", "blocks": [{"n_snapshots": 6, "temperature_K": 300}]},
                            {"recipe_id": "md_b", "blocks": [{"n_snapshots": 9, "temperature_K": 500}]},
                        ],
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)
            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            results = runner.results()["results"]
            self.assertEqual(results["dataset_sweep"]["total_datasets"], 2)
            self.assertEqual(results["dataset_sweep"]["rows"][0]["status"], "dry_run")

    def test_dataset_sweep_generation_enforces_joint_artifact_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,)):
                self.assertEqual(Path(env["PATH"].split(os.pathsep)[0]), Path(command[0]).parent.resolve())
                _write_fake_joint_dataset_from_config(Path(env["PIPELINE_CONFIG_PATH"]), complete=True)
                return {
                    "label": label,
                    "command": command,
                    "cwd": str(cwd),
                    "started_at": 1.0,
                    "finished_at": 2.0,
                    "elapsed_seconds": 1.0,
                    "returncode": 0,
                }

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            runner.start(
                {
                    "dataset_root": str(Path(tmp) / "datasets"),
                    "output_root": str(Path(tmp) / "results"),
                    "dataset_mode": "generate_new",
                    "run_mode": "generate_datasets_only",
                    "split_mode": "block",
                    "dataset_sweep": {
                        "enabled": True,
                        "recipes": [{"recipe_id": "md_valid", "blocks": [{"n_snapshots": 6, "temperature_K": 300}]}],
                    },
                }
            )
            runner._thread.join(timeout=5)

            self.assertEqual(runner.status()["returncode"], 0)
            row = runner.results()["results"]["dataset_sweep"]["rows"][0]
            self.assertEqual(row["artifact_validation_status"], "valid")
            self.assertTrue(Path(row["benchmark_manifest_path"]).exists())

    def test_dataset_sweep_rejects_old_graph2mat_only_generation_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,)):
                _write_fake_joint_dataset_from_config(Path(env["PIPELINE_CONFIG_PATH"]), complete=False)
                return {
                    "label": label,
                    "command": command,
                    "cwd": str(cwd),
                    "started_at": 1.0,
                    "finished_at": 2.0,
                    "elapsed_seconds": 1.0,
                    "returncode": 0,
                }

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            runner.start(
                {
                    "dataset_root": str(Path(tmp) / "datasets"),
                    "output_root": str(Path(tmp) / "results"),
                    "dataset_mode": "generate_new",
                    "run_mode": "generate_datasets_only",
                    "split_mode": "block",
                    "dataset_sweep": {
                        "enabled": True,
                        "recipes": [{"recipe_id": "md_incomplete", "blocks": [{"n_snapshots": 6, "temperature_K": 300}]}],
                    },
                }
            )
            runner._thread.join(timeout=5)

            self.assertEqual(runner.status()["returncode"], 1)
            self.assertIn("missing", runner.status()["error"])

    def test_training_sweep_rejects_generate_new_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()
            with self.assertRaisesRegex(RuntimeError, "Generate and validate datasets first"):
                runner.start(
                    {
                        "dataset_root": str(Path(tmp) / "datasets"),
                        "output_root": str(Path(tmp) / "results"),
                        "dataset_mode": "generate_new",
                        "snapshot_count": 6,
                        "training_sweep": {"enabled": True, "common": {"epochs": [1]}},
                    }
                )

    def test_full_strict_pipeline_requires_training_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()
            with self.assertRaisesRegex(RuntimeError, "training_sweep.enabled=true"):
                runner.start(
                    {
                        "dataset_root": str(Path(tmp) / "datasets"),
                        "output_root": str(Path(tmp) / "results"),
                        "dataset_mode": "full_strict_pipeline",
                        "dry_run": True,
                        "dataset_sweep": {
                            "enabled": True,
                            "recipes": [{"recipe_id": "md_small", "blocks": [{"n_snapshots": 6, "temperature_K": 300}]}],
                        },
                        "training_sweep": {"enabled": False},
                    }
                )

    def test_full_strict_pipeline_dry_run_plans_generation_and_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()
            calls = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,)):
                calls.append(command)
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            status = runner.start(
                {
                    "dataset_root": str(Path(tmp) / "datasets"),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "full_strict_dry",
                    "dataset_mode": "full_strict_pipeline",
                    "run_mode": "full_strict_pipeline",
                    "dry_run": True,
                    "split_mode": "block",
                    "dataset_sweep": {
                        "enabled": True,
                        "recipes": [
                            {"recipe_id": "md_small", "blocks": [{"n_snapshots": 6, "temperature_K": 300}]}
                        ],
                    },
                    "training_sweep": {
                        "enabled": True,
                        "common": {"epochs": [1], "learning_rate": [0.001], "batch_size": [1], "seeds": [42]},
                        "graph2mat": {"enabled": True, "max_ell": [2], "hidden_irreps_channels": [4]},
                        "deeph": {"enabled": False},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            self.assertEqual(calls, [])
            results = runner.results()["results"]
            self.assertEqual(results["dataset_sweep"]["run_mode"], "full_strict_pipeline")
            self.assertEqual(results["dataset_sweep"]["rows"][0]["status"], "dry_run")
            self.assertEqual(results["training_sweep"]["status"], "planned_dry_run")
            self.assertEqual(len(results["training_sweep"]["planned_runs"]), 1)
            self.assertTrue(
                (Path(tmp) / "results" / "full_strict_dry" / "sweep" / "training_sweep_manifest.json").exists()
            )

    def test_training_sweep_dry_run_plans_graph2mat_and_deeph_without_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()
            calls = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,)):
                calls.append(command)
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "training_sweep_dry",
                    "dry_run": True,
                    "performance": {"max_parallel_graph2mat_training_jobs": 2},
                    "training_sweep": {
                        "enabled": True,
                        "common": {"epochs": [1], "learning_rate": [0.001], "batch_size": [1], "seeds": [42]},
                        "graph2mat": {"enabled": True, "max_ell": [2], "hidden_irreps_channels": [4]},
                        "deeph": {"enabled": True, "atom_fea_len": [64], "num_l": [5]},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            self.assertEqual(final["training_sweep"]["completed"], 2)
            self.assertIn("ranking", final["phases"])
            self.assertEqual(calls, [])
            results = runner.results()["results"]
            self.assertEqual(len(results["training_sweep"]["runs"]), 2)
            self.assertEqual(results["training_sweep"]["graph2mat_parallelism"], 2)
            self.assertEqual(results["training_sweep"]["deeph_parallelism"], 1)
            self.assertEqual(results["training_sweep"]["metric_fail_policy"], "fail_closed")
            self.assertNotIn("derivative_workflows", results["training_sweep"])
            deeph_run = next(row for row in results["training_sweep"]["runs"] if row["model"] == "deeph")
            deeph_manifest = json.loads(Path(deeph_run["deeph_manifest_path"]).read_text(encoding="utf-8"))
            self.assertIn(deeph_manifest["deeph_discovery_source"], {"config", "env", "PATH", "sibling_repo", "unavailable"})
            self.assertIn("deeph_discovery", deeph_manifest)
            self.assertEqual(results["ranking"]["recommendation"]["status"], "invalid_incomplete_grid")
            self.assertTrue((Path(tmp) / "results" / "training_sweep_dry" / "sweep" / "training_sweep_manifest.json").exists())
            self.assertTrue((Path(tmp) / "results" / "training_sweep_dry" / "sweep" / "search_plan.json").exists())
            self.assertTrue((Path(tmp) / "results" / "training_sweep_dry" / "sweep" / "budget_summary.json").exists())
            self.assertTrue((Path(tmp) / "results" / "training_sweep_dry" / "summary" / "ranking" / "ranking_summary.json").exists())

    def test_equal_gpu_hours_budget_skips_new_trials_after_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            def fake_graph2mat_job(payload, validation, record):
                return {
                    **record,
                    "status": "completed",
                    "run_root": str(Path(tmp) / "runs" / record["config_id"]),
                    "telemetry": {"gpu_hours_total": 0.1},
                }

            runner._run_training_sweep_graph2mat_job = fake_graph2mat_job  # type: ignore[method-assign]
            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "budget_gpu",
                    "benchmark_mode": "final_publication",
                    "training_sweep": {
                        "enabled": True,
                        "budget_policy": {"mode": "equal_gpu_hours_per_model", "gpu_hours_per_model": 0.15},
                        "common": {"epochs": [1], "learning_rate": [0.001], "batch_size": [1], "seeds": [42]},
                        "graph2mat": {"enabled": True, "num_interactions": [1, 2, 3]},
                        "deeph": {"enabled": False},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            results = runner.results()["results"]
            summary = results["training_sweep"]
            completed = [row for row in summary["runs"] if row["status"] == "completed"]
            skipped = [row for row in summary["runs"] if row["status"] == "skipped_budget_exhausted"]
            self.assertEqual(len(completed), 2)
            self.assertEqual(len(skipped), 1)
            self.assertAlmostEqual(summary["budget"]["consumed_gpu_hours_by_model"]["graph2mat"], 0.2)
            self.assertEqual(summary["budget"]["skipped_trials_by_model"]["graph2mat"], 1)
            self.assertTrue((Path(tmp) / "results" / "budget_gpu" / "sweep" / "budget_summary.json").exists())

    def test_equal_gpu_hours_missing_telemetry_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            def fake_graph2mat_job(payload, validation, record):
                return {
                    **record,
                    "status": "completed",
                    "run_root": str(Path(tmp) / "runs" / record["config_id"]),
                }

            runner._run_training_sweep_graph2mat_job = fake_graph2mat_job  # type: ignore[method-assign]
            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "budget_missing_telemetry",
                    "benchmark_mode": "final_publication",
                    "training_sweep": {
                        "enabled": True,
                        "budget_policy": {"mode": "equal_gpu_hours_per_model", "gpu_hours_per_model": 1.0},
                        "common": {"epochs": [1]},
                        "graph2mat": {"enabled": True, "num_interactions": [1]},
                        "deeph": {"enabled": False},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 1)
            results = runner.results()["results"]
            self.assertIn("Missing gpu_hours_total", results["training_sweep"]["failed_runs"][0]["error"])
            self.assertEqual(results["training_sweep"]["budget"]["budget_accounting_status"], "failed")

    def test_exploratory_equal_gpu_hours_missing_telemetry_records_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            def fake_graph2mat_job(payload, validation, record):
                return {
                    **record,
                    "status": "completed",
                    "run_root": str(Path(tmp) / "runs" / record["config_id"]),
                }

            runner._run_training_sweep_graph2mat_job = fake_graph2mat_job  # type: ignore[method-assign]
            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "budget_missing_telemetry_exploratory",
                    "benchmark_mode": "exploratory_weekend_fast",
                    "training_sweep": {
                        "enabled": True,
                        "budget_policy": {"mode": "equal_gpu_hours_per_model", "gpu_hours_per_model": 1.0},
                        "common": {"epochs": [1]},
                        "graph2mat": {"enabled": True, "num_interactions": [1]},
                        "deeph": {"enabled": False},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            results = runner.results()["results"]
            summary = results["training_sweep"]
            self.assertEqual(summary["failed_runs"], [])
            self.assertIn("Missing gpu_hours_total", summary["warnings"][0])
            self.assertEqual(summary["runs"][0]["budget_accounting_status"], "incomplete")
            self.assertEqual(summary["budget"]["budget_accounting_status"], "failed")

    def test_final_benchmark_training_sweep_locks_test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()
            calls = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,)):
                calls.append(command)
                return {"label": label, "returncode": 0}

            runner._run_command = fake_run_command  # type: ignore[method-assign]
            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "paper_search_dry",
                    "dry_run": True,
                    "benchmark_mode": "final_publication",
                    "training_sweep": {
                        "enabled": True,
                        "common": {"epochs": [1], "learning_rate": [0.001], "batch_size": [1], "seeds": [42]},
                        "graph2mat": {"enabled": True, "max_ell": [2], "hidden_irreps_channels": [4]},
                        "deeph": {"enabled": False},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            self.assertEqual(calls, [])
            results = runner.results()["results"]
            self.assertIsNone(results["ranking"])
            self.assertTrue(results["test_blindness"]["final_test_locked"])
            self.assertEqual(results["test_blindness"]["protocol_stage"], "search")
            run = results["training_sweep"]["runs"][0]
            self.assertEqual(run["protocol_stage"], "search")
            self.assertTrue(run["test_metrics_locked"])
            self.assertEqual(run["test_metrics_status"], "locked_until_final")
            self.assertNotIn("predict_run", run)
            self.assertNotIn("metrics_run", run)
            self.assertTrue(
                (Path(tmp) / "results" / "paper_search_dry" / "summary" / "test_blindness_manifest.json").exists()
            )

    def test_final_benchmark_search_writes_topk_and_robust_rerun_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            def fake_graph2mat_job(payload, validation, record):
                value = 0.1 if int(record["index"]) == 2 else 0.2
                return {
                    **record,
                    "status": "completed",
                    "run_root": str(Path(tmp) / "runs" / record["config_id"]),
                    "metric_split": "validation",
                    "early_stopping": {
                        "validation_metric_name": "val_loss",
                        "best_validation_value": value,
                    },
                    "telemetry": {"gpu_hours_total": 0.01},
                }

            def fake_deeph_job(payload, validation, record):
                value = 0.15 if int(record["index"]) == 4 else 0.3
                return {
                    **record,
                    "status": "completed",
                    "run_root": str(Path(tmp) / "runs" / record["config_id"]),
                    "metric_split": "validation",
                    "early_stopping": {
                        "validation_metric_name": "val_loss",
                        "best_validation_value": value,
                    },
                    "telemetry": {"gpu_hours_total": 0.02},
                }

            runner._run_training_sweep_graph2mat_job = fake_graph2mat_job  # type: ignore[method-assign]
            runner._run_training_sweep_deeph_job = fake_deeph_job  # type: ignore[method-assign]
            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "paper_search_topk",
                    "benchmark_mode": "final_publication",
                    "final_seeds": [0, 1, 2],
                    "top_k_selection": {"k_per_model": 1, "split": "validation", "metric": "val_loss"},
                    "training_sweep": {
                        "enabled": True,
                        "common": {"epochs": [1], "learning_rate": [0.001], "batch_size": [1], "seeds": [42]},
                        "graph2mat": {"enabled": True, "max_ell": [2], "hidden_irreps_channels": [4, 8]},
                        "deeph": {"enabled": True, "atom_fea_len": [64, 128], "edge_fea_len": [64]},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            results = runner.results()["results"]
            selection = results["selection"]
            self.assertEqual(selection["status"], "planned")
            selected = selection["selected_configs"]["selected_configs"]
            selected_by_model = {row["model"]: row for row in selected}
            self.assertEqual(selected_by_model["graph2mat"]["validation_metric_value"], 0.1)
            self.assertEqual(selected_by_model["graph2mat"]["index"], 2)
            self.assertEqual(selected_by_model["deeph"]["validation_metric_value"], 0.15)
            self.assertEqual(selected_by_model["deeph"]["index"], 4)
            plan = selection["robust_rerun_plan"]
            self.assertEqual(plan["planned_run_count"], 6)
            self.assertTrue((Path(tmp) / "results" / "paper_search_topk" / "summary" / "selection" / "selected_configs.json").exists())
            self.assertTrue((Path(tmp) / "results" / "paper_search_topk" / "summary" / "selection" / "robust_rerun_plan.json").exists())

    def test_available_datasets_payload_lists_ready_joint_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets" / "joint_a"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            payload = runner.available_datasets_payload(root=Path(tmp) / "datasets")

            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["ready_count"], 1)
            item = payload["datasets"][0]
            self.assertEqual(item["dataset_root"], str(dataset))
            self.assertTrue(item["benchmark_ready"])
            self.assertEqual(item["total_snapshots"], 3)

    def test_graph2mat_context_applies_payload_performance_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()
            validation = runner.validate_dataset_payload({"dataset_root": str(dataset)})

            context = runner._prepare_graph2mat_context(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "gpu_perf",
                    "performance": {
                        "preset": "aggressive",
                        "compute_accelerator": "gpu",
                        "batch_size": 64,
                        "store_in_memory": True,
                        "torch_num_threads": 8,
                        "torch_float32_matmul_precision": "high",
                        "torch_mixed_precision": "bf16-mixed",
                        "graph2mat_log_every_n_steps": 10,
                        "graph2mat_check_val_every_n_epoch": 5,
                        "graph2mat_checkpoint_every_n_epochs": 5,
                    },
                },
                validation,
            )

            config = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["training"]["trainer"]["accelerator"], "gpu")
            self.assertEqual(config["training"]["trainer"]["precision"], "bf16-mixed")
            self.assertEqual(config["training"]["trainer"]["log_every_n_steps"], 10)
            self.assertEqual(config["training"]["trainer"]["check_val_every_n_epoch"], 5)
            callbacks = config["training"]["trainer"]["callbacks"]
            self.assertEqual(len(callbacks), 2)
            self.assertEqual(callbacks[0]["init_args"]["every_n_epochs"], 5)
            self.assertEqual(callbacks[1]["init_args"]["every_n_epochs"], 5)
            self.assertEqual(config["training"]["data"]["batch_size"], 64)
            self.assertTrue(config["training"]["data"]["store_in_memory"])
            self.assertEqual(config["training"]["torch_float32_matmul_precision"], "high")
            self.assertEqual(config["training"]["data"]["runs_json"], "runs.json")
            self.assertNotIn("train_runs", config["training"]["data"])
            self.assertNotIn("val_runs", config["training"]["data"])
            self.assertEqual(context.runs_json_counts["train"], 1)
            self.assertEqual(context.runs_json_counts["val"], 1)
            manifest = json.loads(context.graph2mat_manifest_path.read_text(encoding="utf-8"))
            accounting = manifest["extra"]["optimizer_update_accounting"]
            self.assertEqual(accounting["train_samples"], 1)
            self.assertEqual(accounting["batch_size"], 64)
            self.assertEqual(accounting["steps_per_epoch"], 1)
            self.assertEqual(accounting["total_optimizer_updates"], accounting["max_epochs"])

    def test_graph2mat_cuequivariance_requirement_fails_early_when_unavailable(self):
        runner = Graph2MatDeepHBenchmarkRunner()
        runner._graph2mat_acceleration_status = lambda _python: {  # type: ignore[method-assign]
            "acceleration_available": False,
            "cuequivariance_available": False,
            "cuequivariance_torch_available": False,
        }

        with self.assertRaisesRegex(RuntimeError, "graph2mat_require_cuequivariance=true"):
            runner._enforce_graph2mat_acceleration_policy(
                {"performance": {"graph2mat_require_cuequivariance": True}},
                "python3",
            )
            runs_json = json.loads(context.runs_json_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(runs_json), ["predict", "test", "train", "val"])
            self.assertFalse(Path(runs_json["train"][0]).is_absolute())

    def test_graph2mat_training_parallelism_comes_from_performance_payload(self):
        runner = Graph2MatDeepHBenchmarkRunner()
        self.assertEqual(runner.status()["training_sweep"], {})
        from g2m_deeph_runner import _graph2mat_training_parallelism

        self.assertEqual(
            _graph2mat_training_parallelism({"performance": {"max_parallel_graph2mat_training_jobs": "2"}}),
            2,
        )

    def test_deeph_training_parallelism_comes_from_performance_payload(self):
        self.assertEqual(
            _deeph_training_parallelism({"performance": {"max_parallel_deeph_training_jobs": "2"}}),
            2,
        )

    def test_deeph_uses_gpu_when_performance_requests_gpu(self):
        disable_cuda, device = _deeph_device_settings(
            {"performance": {"compute_accelerator": "gpu"}},
            {},
        )
        self.assertFalse(disable_cuda)
        self.assertEqual(device, "cuda:0")

        disable_cuda, device = _deeph_device_settings(
            {"performance": {"compute_accelerator": "gpu"}},
            {"disable_cuda": True},
        )
        self.assertTrue(disable_cuda)
        self.assertEqual(device, "cpu")

    def test_training_sweep_dry_run_batches_deeph_jobs_in_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "deeph_parallel_dry",
                    "dry_run": True,
                    "performance": {"max_parallel_deeph_training_jobs": 2},
                    "training_sweep": {
                        "enabled": True,
                        "common": {"epochs": [1], "learning_rate": [0.001], "batch_size": [1], "seeds": [42]},
                        "graph2mat": {"enabled": False},
                        "deeph": {"enabled": True, "atom_fea_len": [64, 128], "num_l": [5]},
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            results = runner.results()["results"]
            runs = results["training_sweep"]["runs"]
            self.assertEqual(len(runs), 2)
            self.assertEqual(results["training_sweep"]["deeph_parallelism"], 2)
            self.assertIn("Running DeepH sweep batch", "".join(runner.logs(since=0)["lines"]))
            run_roots = {row["run_root"] for row in runs}
            self.assertEqual(len(run_roots), 2)
            manifest_paths = {row["deeph_manifest_path"] for row in runs}
            self.assertEqual(len(manifest_paths), 2)
            deeph_roots = []
            for row in runs:
                manifest = json.loads(Path(row["deeph_manifest_path"]).read_text(encoding="utf-8"))
                context = manifest["context"]
                deeph_roots.append(context["root"])
                for key in ("raw_dir", "processed_dir", "graph_dir", "save_dir", "inference_dir", "manifest_path"):
                    self.assertIn(key, context)
            self.assertEqual(len(set(deeph_roots)), 2)

    def test_training_sweep_dry_run_batches_mixed_models_in_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "mixed_parallel_dry",
                    "dry_run": True,
                    "performance": {
                        "max_parallel_graph2mat_training_jobs": 2,
                        "max_parallel_deeph_training_jobs": 1,
                        "mixed_model_training_batches": True,
                    },
                    "training_sweep": {
                        "enabled": True,
                        "search_policy": {"strategy": "manual"},
                        "manual_runs": [
                            {
                                "model": "graph2mat",
                                "config_id": "G2M-1",
                                "overrides": {"max_ell": 2, "hidden_irreps_channels": 4, "max_epochs": 1},
                            },
                            {
                                "model": "deeph",
                                "config_id": "DH-1",
                                "overrides": {"atom_fea_len": 64, "num_l": 5, "epochs": 1},
                            },
                            {
                                "model": "graph2mat",
                                "config_id": "G2M-2",
                                "overrides": {"max_ell": 2, "hidden_irreps_channels": 8, "max_epochs": 1},
                            },
                        ],
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            results = runner.results()["results"]
            self.assertTrue(results["training_sweep"]["mixed_model_training_batches"])
            self.assertEqual(len(results["training_sweep"]["runs"]), 3)
            self.assertIn("Running mixed Graph2Mat/DeepH sweep batch", "".join(runner.logs(since=0)["lines"]))

    def test_training_sweep_can_reorder_into_alternating_model_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            runner = Graph2MatDeepHBenchmarkRunner()

            status = runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "run_id": "alternating_model_batches_dry",
                    "dry_run": True,
                    "performance": {
                        "max_parallel_graph2mat_training_jobs": 2,
                        "max_parallel_deeph_training_jobs": 1,
                        "model_batch_schedule": "alternating",
                    },
                    "training_sweep": {
                        "enabled": True,
                        "search_policy": {"strategy": "manual"},
                        "manual_runs": [
                            {
                                "model": "graph2mat",
                                "config_id": "G2M-1",
                                "overrides": {"max_ell": 2, "hidden_irreps_channels": 4, "max_epochs": 1},
                            },
                            {
                                "model": "deeph",
                                "config_id": "DH-1",
                                "overrides": {"atom_fea_len": 64, "num_l": 5, "epochs": 1},
                            },
                            {
                                "model": "graph2mat",
                                "config_id": "G2M-2",
                                "overrides": {"max_ell": 2, "hidden_irreps_channels": 8, "max_epochs": 1},
                            },
                            {
                                "model": "deeph",
                                "config_id": "DH-2",
                                "overrides": {"atom_fea_len": 128, "num_l": 5, "epochs": 1},
                            },
                            {
                                "model": "graph2mat",
                                "config_id": "G2M-3",
                                "overrides": {"max_ell": 2, "hidden_irreps_channels": 16, "max_epochs": 1},
                            },
                        ],
                    },
                }
            )
            self.assertTrue(status["running"])
            runner._thread.join(timeout=5)

            final = runner.status()
            self.assertFalse(final["running"])
            self.assertEqual(final["returncode"], 0)
            summary = runner.results()["results"]["training_sweep"]
            self.assertEqual(summary["model_batch_schedule"], "alternating")
            self.assertEqual(
                [row["config_id"] for row in summary["planned_runs"]],
                ["G2M-1", "G2M-2", "DH-1", "G2M-3", "DH-2"],
            )
            logs = "".join(runner.logs(since=0)["lines"])
            self.assertIn("Running Graph2Mat sweep batch: G2M-1, G2M-2", logs)
            self.assertNotIn("Running mixed Graph2Mat/DeepH sweep batch", logs)

    def test_training_sweep_derivative_postprocess_runs_for_completed_child_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "benchmark"
            child_a = run_root / "sweep" / "graph2mat" / "dataset" / "g2m"
            child_b = run_root / "sweep" / "deeph" / "dataset" / "deeph"
            (child_a / "metrics" / "graph2mat" / "eval_input").mkdir(parents=True)
            (child_b / "metrics" / "deeph" / "eval").mkdir(parents=True)
            summary = {
                "runs": [
                    {"status": "completed", "run_root": str(child_a), "config_id": "g2m"},
                    {"status": "completed", "run_root": str(child_b), "config_id": "deeph"},
                ]
            }
            payload = {
                "workflow_mode": "h_then_derivative_postprocess",
                "derivative": {"enabled": True, "method": "central"},
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            runner = Graph2MatDeepHBenchmarkRunner()
            calls = []

            def fake_modular(child_payload, *, run_root=None, graph2mat_context=None, deeph_context=None):
                calls.append((child_payload, run_root, graph2mat_context, deeph_context))
                result_dir = Path(run_root) / "derivative_workflow"
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "derivative_workflow_manifest.json").write_text("{}\n", encoding="utf-8")
                return {"result_dir": str(result_dir), "stages": {}}

            runner._run_modular_derivative_workflow = fake_modular  # type: ignore[method-assign]

            records = runner._run_training_sweep_derivative_workflows(payload, run_root=run_root, summary=summary)

            self.assertEqual([call[1] for call in calls], [child_a.resolve(), child_b.resolve()])
            first_derivative = calls[0][0]["modular_workflow"]["derivative"]
            second_derivative = calls[1][0]["modular_workflow"]["derivative"]
            self.assertEqual(first_derivative["graph2mat_result_dir"], str(child_a / "metrics" / "graph2mat" / "eval_input"))
            self.assertEqual(second_derivative["deeph_result_dir"], str(child_b / "metrics" / "deeph" / "eval"))
            self.assertEqual([record["derivative_workflow_status"] for record in records], ["completed", "completed"])
            manifest = json.loads((run_root / "sweep" / "training_sweep_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["derivative_workflows"]), 2)

    def test_training_sweep_derivative_full_schedules_full_stage_set_for_child_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "benchmark"
            child = run_root / "sweep" / "combined" / "dataset" / "run"
            child.mkdir(parents=True)
            dataset_root = root / "child_dataset"
            dataset_root.mkdir()
            graph_context, deeph_context = self._benchmark_contexts(root / "contexts")
            graph_context.graph2mat_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            deeph_context.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            runner = Graph2MatDeepHBenchmarkRunner()
            runner._write_graph2mat_manifest(graph_context, checkpoint_manifest={"checkpoint_path": str(graph_context.training_dir / "best.ckpt")})
            runner._write_deeph_manifest(deeph_context)
            summary = {
                "runs": [
                    {
                        "status": "completed",
                        "run_root": str(child),
                        "config_id": "full",
                        "dataset_root": str(dataset_root),
                        "graph2mat_manifest_path": str(graph_context.graph2mat_manifest_path),
                        "deeph_manifest_path": str(deeph_context.manifest_path),
                    }
                ]
            }
            payload = {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": str(root / "dataset"),
                    "method": "central",
                    "delta_ang": 0.01,
                    "atoms": ["0"],
                    "axes": ["x"],
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            calls = []

            def fake_modular(child_payload, *, run_root=None, graph2mat_context=None, deeph_context=None):
                calls.append((child_payload, run_root, graph2mat_context, deeph_context))
                result_dir = Path(run_root) / "derivative_workflow"
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "derivative_workflow_manifest.json").write_text("{}\n", encoding="utf-8")
                return {"result_dir": str(result_dir), "stages": {}}

            runner._run_modular_derivative_workflow = fake_modular  # type: ignore[method-assign]

            records = runner._run_training_sweep_derivative_workflows(payload, run_root=run_root, summary=summary)

            stages = calls[0][0]["modular_workflow"]["stages"]
            for stage in (
                "build_derivative_stencils",
                "validate_derivative_stencils",
                "run_derivative_siesta_reference",
                "predict_derivative_graph2mat",
                "predict_derivative_deeph",
                "derivative_metrics_graph2mat",
                "derivative_metrics_deeph",
                "derivative_gate_check",
                "derivative_plots",
            ):
                self.assertTrue(stages[stage])
            self.assertIsNotNone(calls[0][2])
            self.assertIsNotNone(calls[0][3])
            self.assertEqual(
                calls[0][0]["modular_workflow"]["derivative"]["source_dataset_root"],
                str(dataset_root.resolve()),
            )
            self.assertEqual(records[0]["derivative_workflow_status"], "completed")

    def test_training_sweep_derivative_full_uses_child_dataset_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "benchmark"
            child_a = run_root / "sweep" / "combined" / "dataset_a" / "run"
            child_b = run_root / "sweep" / "combined" / "dataset_b" / "run"
            dataset_a = root / "datasets" / "a"
            dataset_b = root / "datasets" / "b"
            child_a.mkdir(parents=True)
            child_b.mkdir(parents=True)
            dataset_a.mkdir(parents=True)
            dataset_b.mkdir(parents=True)
            summary = {
                "runs": [
                    {"status": "completed", "run_root": str(child_a), "config_id": "a", "dataset_root": str(dataset_a)},
                    {"status": "completed", "run_root": str(child_b), "config_id": "b", "dataset_root": str(dataset_b)},
                ]
            }
            payload = {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": str(root / "global_dataset"),
                    "method": "central",
                    "delta_ang": 0.01,
                    "atoms": ["0"],
                    "axes": ["x"],
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            runner = Graph2MatDeepHBenchmarkRunner()
            calls = []

            def fake_modular(child_payload, *, run_root=None, graph2mat_context=None, deeph_context=None):
                calls.append((child_payload, run_root))
                result_dir = Path(run_root) / "derivative_workflow"
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "derivative_workflow_manifest.json").write_text("{}\n", encoding="utf-8")
                return {"result_dir": str(result_dir), "stages": {}}

            runner._run_modular_derivative_workflow = fake_modular  # type: ignore[method-assign]

            records = runner._run_training_sweep_derivative_workflows(payload, run_root=run_root, summary=summary)

            self.assertEqual([record["derivative_workflow_status"] for record in records], ["completed", "completed"])
            self.assertEqual(
                [call[0]["modular_workflow"]["derivative"]["source_dataset_root"] for call in calls],
                [str(dataset_a.resolve()), str(dataset_b.resolve())],
            )

    def test_training_sweep_derivative_full_child_payloads_pass_preflight_with_default_siesta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "benchmark"
            child_a = run_root / "sweep" / "combined" / "dataset_a" / "run"
            child_b = run_root / "sweep" / "combined" / "dataset_b" / "run"
            dataset_a = root / "datasets" / "a"
            dataset_b = root / "datasets" / "b"
            child_a.mkdir(parents=True)
            child_b.mkdir(parents=True)
            dataset_a.mkdir(parents=True)
            dataset_b.mkdir(parents=True)
            summary = {
                "runs": [
                    {"status": "completed", "run_root": str(child_a), "config_id": "a", "dataset_root": str(dataset_a)},
                    {"status": "completed", "run_root": str(child_b), "config_id": "b", "dataset_root": str(dataset_b)},
                ]
            }
            payload = {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": str(root / "global_dataset"),
                    "method": "central",
                    "delta_ang": 0.01,
                    "atoms": ["0"],
                    "axes": ["x"],
                    "skip_if_exists": False,
                },
                "stages": {
                    "build_derivative_stencils": True,
                    "validate_derivative_stencils": False,
                    "run_derivative_siesta_reference": True,
                    "predict_derivative_graph2mat": False,
                    "predict_derivative_deeph": False,
                    "derivative_metrics_graph2mat": False,
                    "derivative_metrics_deeph": False,
                    "derivative_gate_check": False,
                    "derivative_plots": False,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            runner = Graph2MatDeepHBenchmarkRunner()
            preflighted_source_roots = []

            def fake_modular(child_payload, *, run_root=None, graph2mat_context=None, deeph_context=None):
                workflow = child_payload["modular_workflow"]
                config = workflow["derivative"]
                self.assertNotIn("siesta_command", config)
                runner._preflight_derivative_workflow(
                    child_payload,
                    stages=workflow["stages"],
                    config=config,
                    common_root=runner._derivative_root(child_payload, run_root=run_root),
                    run_root=run_root,
                    graph2mat_context=graph2mat_context,
                    deeph_context=deeph_context,
                )
                preflighted_source_roots.append(config["source_dataset_root"])
                result_dir = Path(run_root) / "derivative_workflow"
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "derivative_workflow_manifest.json").write_text("{}\n", encoding="utf-8")
                return {"result_dir": str(result_dir), "stages": {}}

            runner._run_modular_derivative_workflow = fake_modular  # type: ignore[method-assign]

            records = runner._run_training_sweep_derivative_workflows(payload, run_root=run_root, summary=summary)

            self.assertEqual([record["derivative_workflow_status"] for record in records], ["completed", "completed"])
            self.assertEqual(preflighted_source_roots, [str(dataset_a.resolve()), str(dataset_b.resolve())])

            missing_dataset_records = runner._run_training_sweep_derivative_workflows(
                payload,
                run_root=run_root,
                summary={"runs": [{"status": "completed", "run_root": str(child_a), "config_id": "missing_dataset"}]},
            )
            self.assertEqual(missing_dataset_records[0]["derivative_workflow_status"], "failed")
            self.assertIn("dataset_root", missing_dataset_records[0]["failure_reason"])
            self.assertIn("build_derivative_stencils", missing_dataset_records[0]["failure_reason"])

    def test_training_sweep_derivative_full_missing_child_dataset_root_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "benchmark"
            child = run_root / "sweep" / "combined" / "dataset" / "run"
            child.mkdir(parents=True)
            summary = {"runs": [{"status": "completed", "run_root": str(child), "config_id": "full"}]}
            payload = {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "source_dataset_root": str(root / "global_dataset"),
                    "method": "central",
                    "delta_ang": 0.01,
                    "atoms": ["0"],
                    "axes": ["x"],
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            runner = Graph2MatDeepHBenchmarkRunner()

            records = runner._run_training_sweep_derivative_workflows(payload, run_root=run_root, summary=summary)

            self.assertEqual(records[0]["derivative_workflow_status"], "failed")
            self.assertIn("dataset_root", records[0]["failure_reason"])
            self.assertIn("build_derivative_stencils", records[0]["failure_reason"])

    def test_training_sweep_derivative_full_missing_model_artifact_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "benchmark"
            child = run_root / "sweep" / "graph2mat" / "dataset" / "g2m"
            child.mkdir(parents=True)
            (child / "derivatives").mkdir()
            summary = {"runs": [{"status": "completed", "run_root": str(child), "config_id": "g2m"}]}
            payload = {
                "workflow_mode": "h_then_derivative_full",
                "derivative": {
                    "enabled": True,
                    "result_dir": str(child / "derivatives"),
                    "method": "central",
                },
                "stages": {
                    "build_derivative_stencils": False,
                    "validate_derivative_stencils": False,
                    "run_derivative_siesta_reference": False,
                    "predict_derivative_graph2mat": True,
                    "predict_derivative_deeph": False,
                    "derivative_metrics_graph2mat": False,
                    "derivative_metrics_deeph": False,
                    "derivative_gate_check": False,
                    "derivative_plots": False,
                },
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
            runner = Graph2MatDeepHBenchmarkRunner()

            records = runner._run_training_sweep_derivative_workflows(payload, run_root=run_root, summary=summary)

            self.assertEqual(records[0]["derivative_workflow_status"], "failed")
            self.assertIn("derivative.graph2mat_checkpoint", records[0]["failure_reason"])
            manifest = json.loads((run_root / "sweep" / "training_sweep_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["derivative_workflows"][0]["derivative_workflow_status"], "failed")

    def test_full_strict_pipeline_passes_derivative_request_to_training_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Graph2MatDeepHBenchmarkRunner()
            runner._state.run_root = str(Path(tmp) / "results" / "strict")
            dataset = Path(tmp) / "datasets" / "md_small"
            captured = {}

            def fake_dataset_sweep(_payload, _sweep_info, *, run_root):
                return {"rows": [{"dataset_root": str(dataset), "dataset_size": 1, "status": "benchmark_ready"}]}

            def fake_training_sweep(payload, validation, plan):
                captured["payload"] = payload
                captured["validation"] = validation
                captured["plan"] = plan

            runner._run_dataset_sweep_generation = fake_dataset_sweep  # type: ignore[method-assign]
            runner._run_training_sweep = fake_training_sweep  # type: ignore[method-assign]
            payload = {
                "workflow_mode": "h_then_derivative_postprocess",
                "run_mode": "full_strict_pipeline",
                "dataset_sweep": {
                    "enabled": True,
                    "recipes": [{"recipe_id": "md_small", "blocks": [{"n_snapshots": 6, "temperature_K": 300}]}],
                },
                "training_sweep": {
                    "enabled": True,
                    "common": {"epochs": [1]},
                    "graph2mat": {"enabled": True, "num_interactions": [1]},
                    "deeph": {"enabled": False},
                },
                "derivative": {"enabled": True, "method": "central"},
            }
            payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)

            runner._run_full_strict_pipeline(payload, {"md_dataset_specs": [{"dataset_slug": "md_small"}]})

            self.assertTrue(captured["payload"]["modular_workflow"]["stages"]["derivative_metrics_graph2mat"])
            self.assertEqual(captured["validation"]["dataset_root"], str(dataset))

    def test_graph2mat_acceleration_probe_is_recorded_without_requiring_optional_package(self):
        runner = Graph2MatDeepHBenchmarkRunner()

        status = runner._graph2mat_acceleration_status(sys.executable)

        self.assertEqual(status["source"], "python_import_probe")
        self.assertIn("cuequivariance_available", status)
        self.assertIn("cuequivariance_torch_available", status)
        self.assertIn("acceleration_available", status)

    def test_link_or_copy_file_is_idempotent_when_destination_already_points_to_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "material_basis" / "C.ion.xml"
            destination = Path(tmp) / "MD_steps" / "0" / "C.ion.xml"
            source.parent.mkdir(parents=True)
            destination.parent.mkdir(parents=True)
            source.write_text("<basis />\n", encoding="utf-8")
            os.symlink(os.path.relpath(source, destination.parent), destination)

            _link_or_copy_file(source, destination)

            self.assertTrue(destination.exists())
            self.assertEqual(destination.resolve(), source.resolve())

    def test_link_or_copy_file_leaves_equivalent_existing_basis_link_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared_basis = Path(tmp) / "MD_steps" / "basis" / "C.ion.xml"
            material_basis = Path(tmp) / "material_basis" / "C.ion.xml"
            destination = Path(tmp) / "MD_steps" / "0" / "C.ion.xml"
            shared_basis.parent.mkdir(parents=True)
            material_basis.parent.mkdir(parents=True)
            destination.parent.mkdir(parents=True)
            shared_basis.write_text("<basis />\n", encoding="utf-8")
            material_basis.write_text("<basis />\n", encoding="utf-8")
            os.symlink(os.path.relpath(shared_basis, destination.parent), destination)

            _link_or_copy_file(material_basis, destination)

            self.assertTrue(destination.exists())
            self.assertEqual(destination.resolve(), shared_basis.resolve())

    def test_resume_training_sweep_loads_only_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            resume_root = Path(tmp) / "previous"
            manifest = resume_root / "sweep" / "training_sweep_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "model": "graph2mat",
                                "dataset_id": "dataset_a",
                                "config_id": "graph2mat_abc",
                                "status": "completed",
                            },
                            {
                                "model": "deeph",
                                "dataset_id": "dataset_a",
                                "config_id": "deeph_failed",
                                "status": "failed",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runner = Graph2MatDeepHBenchmarkRunner()

            root, completed = runner._load_resume_training_sweep(
                {"resume_from_run_root": str(resume_root)},
                run_root=Path(tmp) / "new",
            )

            self.assertEqual(root, resume_root.resolve())
            self.assertEqual(list(completed), ["graph2mat|dataset_a|graph2mat_abc"])

    def test_resume_dataset_sweep_reconstructs_summary_from_existing_validated_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets_root = Path(tmp) / "datasets"
            _write_training_ready_dataset(datasets_root / "md_ready")
            run_root = Path(tmp) / "results" / "current"
            runner = Graph2MatDeepHBenchmarkRunner()
            runner._logs = []

            manifest = runner._load_resume_dataset_sweep(
                {
                    "dataset_root": str(datasets_root),
                    "resume_from_run_root": str(Path(tmp) / "previous_without_summary"),
                    "dataset_sweep": {
                        "enabled": True,
                        "recipes": [
                            {
                                "recipe_id": "md_ready",
                                "label": "ready dataset",
                                "blocks": [{"n_snapshots": 6, "temperature_K": 300}],
                            }
                        ],
                    },
                },
                run_root=run_root,
            )

            self.assertEqual(manifest["total_datasets"], 1)
            self.assertEqual(manifest["rows"][0]["status"], "benchmark_ready")
            self.assertEqual(
                manifest["rows"][0]["source"],
                "reused_existing_dataset_sweep_without_summary",
            )
            self.assertTrue((run_root / "summary" / "dataset_sweep_summary.json").exists())

    def test_resume_retry_cleans_stale_incomplete_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results"
            child_payload = {
                "output_root": str(output_root),
                "run_id": "top/sweep/deeph/dataset_a/deeph_cfg",
                "reuse_run_root": True,
                "resume_training_sweep": True,
            }
            stale = output_root / "top" / "sweep" / "deeph" / "dataset_a" / "deeph_cfg" / "deeph" / "processed"
            stale.mkdir(parents=True)
            (stale / "old_rc.h5").write_text("stale\n", encoding="utf-8")
            runner = Graph2MatDeepHBenchmarkRunner()

            runner._clean_resume_training_run_root(
                child_payload,
                {"dataset_id": "dataset_a", "config_id": "deeph_cfg"},
            )

            self.assertFalse((output_root / "top" / "sweep" / "deeph" / "dataset_a" / "deeph_cfg").exists())

    def test_run_accepts_explicit_repair_request_but_does_not_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_snapshot(dataset, "0", complete=False)

            runner = Graph2MatDeepHBenchmarkRunner()
            status = runner.start({"dataset_root": str(dataset), "allow_repair": True})
            self.assertTrue(status["running"])
            runner._thread.join(timeout=2)
            final_status = runner.status()
            self.assertFalse(final_status["running"])
            self.assertEqual(final_status["stage"], "generate_or_validate_joint_dataset")
            self.assertEqual(final_status["returncode"], 2)
            self.assertIn("repair mode", final_status["error"])

    def test_stop_updates_runner_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_dataset_provenance(dataset)
            _write_snapshot(dataset, "0", complete=True)

            runner = Graph2MatDeepHBenchmarkRunner()
            runner.start(
                {
                    "dataset_root": str(dataset),
                    "output_root": str(Path(tmp) / "results"),
                    "phase_delay_seconds": 0.05,
                }
            )
            status = runner.stop()
            self.assertTrue(status["stop_requested"])
            runner._thread.join(timeout=2)
            self.assertFalse(runner.status()["running"])

    def test_log_response_is_bounded(self):
        runner = Graph2MatDeepHBenchmarkRunner()
        with runner._lock:
            runner._logs = [f"line {idx}\n" for idx in range(10)]

        payload = runner.logs(since=0, limit=3)

        self.assertEqual(payload["offset"], 10)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["dropped_lines"], 7)
        self.assertLessEqual(len(payload["lines"]), 4)

    def test_plots_endpoint_returns_common_metric_payload(self):
        runner = Graph2MatDeepHBenchmarkRunner()
        with runner._lock:
            runner._dataset_validation = {
                "artifact_summary": {
                    "total_snapshots": 1,
                    "valid_snapshots": 1,
                    "missing_required_counts": {},
                }
            }
            runner._phase_timings = [
                {
                    "phase": "graph2mat_train",
                    "started_at": 1.0,
                    "finished_at": 3.0,
                    "elapsed_seconds": 2.0,
                }
            ]
            runner._last_results = {
                "dataset_sweep": {
                    "rows": [
                        {
                            "recipe_id": "md_1",
                            "dataset_root": "/tmp/md_1",
                            "dataset_size": 10,
                            "generation_seconds": 5.0,
                            "status": "benchmark_ready",
                        }
                    ]
                },
                "training_sweep": {
                    "runs": [
                        {
                            "model": "graph2mat",
                            "dataset_id": "md_1",
                            "dataset_root": "/tmp/md_1",
                            "config_id": "g2m_cfg",
                            "status": "completed",
                            "train_run": {"elapsed_seconds": 20.0},
                            "predict_run": {"elapsed_seconds": 2.0},
                            "metrics_run": {"elapsed_seconds": 1.0},
                        }
                    ]
                },
                "common_metrics": {
                    "status": "valid_reused_joint_dataset",
                    "warnings": [],
                    "summary_rows": [
                        {"method": "graph2mat", "h_mae_eV_mean": 0.2},
                        {"method": "deeph", "h_mae_eV_mean": 0.1},
                    ],
                    "recommendation": {
                        "winner": "deeph",
                        "robust_recommendation": True,
                        "primary_metric": "h_mae_eV_mean",
                    },
                },
                "ranking": {
                    "recommendation": {
                        "status": "exploratory_deeph_win",
                        "scientific_status": "exploratory_only",
                        "winner": "deeph",
                        "primary_metric": "h_mae_eV",
                    },
                    "best_runs_by_model": [],
                    "pairwise_graph2mat_vs_deeph": [],
                    "pareto_accuracy_cost": [],
                },
            }

        payload = runner.plots()

        self.assertTrue(payload["available"])
        self.assertEqual(payload["recommendation"]["winner"], "deeph")
        self.assertEqual(payload["ranking"]["recommendation"]["status"], "exploratory_deeph_win")
        self.assertTrue(payload["plots"])
        self.assertTrue(payload["timing_rows"])
        self.assertTrue(payload["timing_scaling_rows"])
        scaling_plot = next(plot for plot in payload["plots"] if plot["id"] == "timing_scaling")
        self.assertEqual(scaling_plot["kind"], "timing_scaling")
        train_row = next(row for row in payload["timing_scaling_rows"] if row["phase"] == "graph2mat_train")
        self.assertEqual(train_row["dataset_size"], 10)
        self.assertEqual(train_row["elapsed_seconds"], 20.0)

    def test_results_endpoint_uses_current_run_plot_payload_without_archive_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            dataset.mkdir(parents=True, exist_ok=True)
            (dataset / "benchmark_dataset_manifest.json").write_text(
                json.dumps({"samples": ["s0"]}),
                encoding="utf-8",
            )

            run_root = Path(tmp) / "results" / "current_run"
            metrics_dir = run_root / "metrics" / "graph2mat" / "eval_input" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / "manifest.json").write_text(
                json.dumps({"summary": {"kpoint_matrix": {"h_mae_eV": {"mean": 0.123}}}}),
                encoding="utf-8",
            )
            (run_root / "sweep").mkdir(parents=True, exist_ok=True)
            (run_root / "sweep" / "training_sweep_manifest.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "status": "completed",
                                "model": "graph2mat",
                                "dataset_id": "dataset",
                                "dataset_root": str(dataset),
                                "run_root": str(run_root),
                                "config_id": "cfg",
                                "metrics_run": {"returncode": 0},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            runner = Graph2MatDeepHBenchmarkRunner()
            runner._state.run_root = str(run_root)
            runner._last_results = {}

            with mock.patch.object(
                runner,
                "_discover_plot_run_roots_locked",
                side_effect=AssertionError("results() should not scan archive roots"),
            ):
                payload = runner.results()

            self.assertTrue(payload["available"])
            self.assertTrue(payload["plot_payload"]["available"])
            self.assertEqual(payload["plot_payload"]["live_metric_rows"], 0)
            self.assertTrue(payload["plot_payload"]["metric_scaling_rows"])
            self.assertTrue(
                any(
                    row.get("run_id") == "current_run"
                    for row in payload["plot_payload"]["metric_scaling_rows"]
                )
            )
            self.assertEqual(payload["plot_payload"]["archived_timing_runs"], 0)

    def test_plots_endpoint_excludes_running_training_sweep_timing_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            run_root = Path(tmp) / "results" / "live_run"
            (run_root / "sweep").mkdir(parents=True)
            planned = {
                "planned_runs": [
                    {
                        "model": "deeph",
                        "dataset_id": "dataset",
                        "dataset_root": str(dataset),
                        "config_id": "DH-live",
                        "overrides": {"epochs": 12},
                    }
                ]
            }
            (run_root / "sweep" / "training_sweep_manifest.json").write_text(
                json.dumps(planned),
                encoding="utf-8",
            )
            runner = Graph2MatDeepHBenchmarkRunner()
            with runner._lock:
                runner._state.run_root = str(run_root)
                runner._state.dataset_root = str(dataset)
                runner._state.training_sweep_status = {
                    "enabled": True,
                    "active_model": "deeph",
                    "active_dataset": "dataset",
                    "active_config_id": "DH-live",
                    "active_started_at": time.time() - 5.0,
                    "active_runs": [
                        {"model": "deeph", "dataset_id": "dataset", "config_id": "DH-live"}
                    ],
                }

            payload = runner.plots()

            running_rows = [
                row
                for row in payload["timing_scaling_rows"]
                if row.get("source") == "live_training_sweep_status"
            ]
            self.assertEqual(running_rows, [])
            self.assertEqual(payload["live_timing_rows"], 0)

    def test_plots_endpoint_includes_archived_metric_scaling_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results" / "graphene_w90_g2m_deeph_benchmark"
            run_root = output_root / "run_archived"
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            summary_dir = run_root / "common_metrics" / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            (summary_dir / "common_summary.json").write_text(
                json.dumps(
                    {
                        "status": "diagnostic_only",
                        "summary_rows": [
                            {"method": "graph2mat", "h_mae_eV_mean": 0.25},
                            {"method": "deeph", "h_mae_eV_mean": 0.15, "diagnostic_only": True},
                        ],
                        "warnings": [],
                        "recommendation": {"winner": None, "robust_recommendation": False},
                    }
                ),
                encoding="utf-8",
            )
            graph_manifest = {
                "context": {"dataset_root": str(dataset)},
                "extra": {"training_run": {"elapsed_seconds": 3.0}},
            }
            graph_dir = run_root / "graph2mat"
            graph_dir.mkdir(parents=True, exist_ok=True)
            (graph_dir / "graph2mat_manifest.json").write_text(json.dumps(graph_manifest), encoding="utf-8")

            runner = Graph2MatDeepHBenchmarkRunner()
            with mock.patch("g2m_deeph_runner.DEFAULT_OUTPUT_ROOT", output_root):
                payload = runner.plots()

            self.assertTrue(payload["available"])
            self.assertGreaterEqual(payload["archived_runs"], 1)
            self.assertTrue(payload["metric_scaling_rows"])
            metric_plot = next(plot for plot in payload["plots"] if plot["id"] == "metric_scaling_h_mae")
            self.assertEqual(metric_plot["kind"], "metric_scaling")
            self.assertTrue(any(row["dataset_size"] == 3 for row in payload["metric_scaling_rows"]))

    def test_plots_endpoint_filters_selected_archived_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results" / "graphene_w90_g2m_deeph_benchmark"
            dataset = Path(tmp) / "dataset"
            _write_training_ready_dataset(dataset)
            for run_name, metric_value in (("run_one", 0.12), ("run_two", 0.34)):
                run_root = output_root / run_name
                summary_dir = run_root / "common_metrics" / "summary"
                summary_dir.mkdir(parents=True, exist_ok=True)
                (summary_dir / "common_summary.json").write_text(
                    json.dumps(
                        {
                            "status": "diagnostic_only",
                            "summary_rows": [
                                {"method": "graph2mat", "h_mae_eV_mean": metric_value},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                graph_dir = run_root / "graph2mat"
                graph_dir.mkdir(parents=True, exist_ok=True)
                (graph_dir / "graph2mat_manifest.json").write_text(
                    json.dumps(
                        {
                            "context": {"dataset_root": str(dataset)},
                            "extra": {"training_run": {"elapsed_seconds": 3.0}},
                        }
                    ),
                    encoding="utf-8",
                )

            runner = Graph2MatDeepHBenchmarkRunner()
            with mock.patch("g2m_deeph_runner.DEFAULT_OUTPUT_ROOT", output_root):
                runs = runner.plot_runs()["runs"]
                selected = next(run for run in runs if run["run_id"] == "run_two")
                payload = runner.plots(selected_run_ids={selected["id"]})

            self.assertTrue(payload["metric_scaling_rows"])
            self.assertEqual({row["run_id"] for row in payload["metric_scaling_rows"]}, {"run_two"})
            self.assertTrue(payload["timing_scaling_rows"])
            self.assertEqual({row["run_id"] for row in payload["timing_scaling_rows"]}, {"run_two"})
            with mock.patch("g2m_deeph_runner.DEFAULT_OUTPUT_ROOT", output_root):
                empty_payload = runner.plots(selected_run_ids=set())
            self.assertEqual(empty_payload["metric_scaling_rows"], [])
            self.assertEqual(empty_payload["timing_scaling_rows"], [])

    def test_plot_runs_discovers_live_run_from_runner_status_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results" / "graphene_w90_g2m_deeph_benchmark"
            run_root = output_root / "run_live_only"
            run_root.mkdir(parents=True)
            (run_root / "runner_status.json").write_text(
                json.dumps(
                    {
                        "schema": "graph2mat_deeph_benchmark_runner_v1",
                        "status": {
                            "running": True,
                            "stage": "training_sweep",
                            "training_sweep": {
                                "completed": 7,
                                "failed": 0,
                                "total": 20,
                                "active_model": "deeph_parallel",
                                "active_dataset": "graphene_5x2_scale_iid20",
                                "active_runs": [
                                    {
                                        "model": "deeph",
                                        "dataset_id": "graphene_5x2_scale_iid20",
                                        "config_id": "DH-live",
                                    }
                                ],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            runner = Graph2MatDeepHBenchmarkRunner()

            with mock.patch("g2m_deeph_runner.DEFAULT_OUTPUT_ROOT", output_root):
                runs = runner.plot_runs()["runs"]

            self.assertTrue(runs)
            entry = next(run for run in runs if run["run_id"] == "run_live_only")
            self.assertEqual(entry["status"], "training_sweep")
            self.assertEqual(entry["planned_runs"], 20)
            self.assertEqual(entry["completed_runs"], 7)
            self.assertEqual(entry["failed_runs"], 0)
            self.assertEqual(entry["models"], ["deeph"])
            self.assertEqual(entry["dataset_ids"], ["graphene_5x2_scale_iid20"])
            self.assertFalse(entry["has_training_sweep"])
            self.assertFalse(entry["has_metric_rows"])

    def test_logs_and_status_attach_to_detached_running_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results" / "graphene_w90_g2m_deeph_benchmark"
            run_root = output_root / "run_detached"
            log_path = run_root / "sweep" / "deeph" / "dataset" / "cfg" / "deeph" / "train" / "result.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("Epoch #12: Train loss: 0.1234. Val loss: 0.2345. Best val loss: 0.2000\n", encoding="utf-8")
            (run_root / "runner_status.json").write_text(
                json.dumps(
                    {
                        "schema": "graph2mat_deeph_benchmark_runner_v1",
                        "status": {
                            "running": True,
                            "stage": "training_sweep",
                            "run_id": "run_detached",
                            "run_root": str(run_root),
                            "active_processes": 3,
                            "training_sweep": {
                                "enabled": True,
                                "completed": 9,
                                "failed": 0,
                                "total": 20,
                                "active_model": "deeph_parallel",
                                "active_dataset": "graphene_5x2_scale_iid60",
                            },
                            "warnings": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            runner = Graph2MatDeepHBenchmarkRunner()
            with mock.patch.object(runner, "_latest_detached_running_run_root", return_value=run_root):
                status = runner.status()
                logs = runner.logs()

            self.assertTrue(status["running"])
            self.assertEqual(status["run_id"], "run_detached")
            self.assertEqual(status["stage"], "training_sweep")
            self.assertEqual(status["training_sweep"]["completed"], 9)
            self.assertTrue(
                any("attached_to_detached_g2m_deeph_run" in warning for warning in status["warnings"])
            )
            self.assertTrue(logs["lines"])
            joined = "".join(logs["lines"])
            self.assertIn("Watching detached benchmark run", joined)
            self.assertIn("Epoch #12", joined)

    def test_plots_endpoint_includes_live_metric_scaling_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "results" / "graphene_w90_g2m_deeph_benchmark"
            run_root = output_root / "run_live"
            run_root.mkdir(parents=True)
            runner = Graph2MatDeepHBenchmarkRunner()
            runner._state.run_root = str(run_root)
            live_row = {
                "run_id": "run_live",
                "dataset_id": "md_1",
                "dataset_root": "/tmp/md_1",
                "dataset_size": 10,
                "method": "graph2mat",
                "config_id": "g2m_cfg",
                "epoch_label": "100 epochs",
                "metric_key": "h_mae_eV_mean",
                "metric_value": 0.123,
                "scientific_status": "valid",
                "source": "live_training_sweep_metrics",
            }

            with (
                mock.patch("g2m_deeph_runner.DEFAULT_OUTPUT_ROOT", output_root),
                mock.patch("g2m_deeph_runner.live_metric_scaling_rows", return_value=[live_row]),
            ):
                payload = runner.plots()

            self.assertTrue(payload["available"])
            self.assertIn(live_row, payload["metric_scaling_rows"])
            self.assertTrue(any(row.get("run_id") == "run_live" for row in payload["metric_scaling_rows"]))
            self.assertEqual(payload["live_metric_rows"], 1)
            self.assertGreaterEqual(payload["archived_runs"], 1)
            metric_plot = next(plot for plot in payload["plots"] if plot["id"] == "metric_scaling_h_mae")
            self.assertEqual(metric_plot["kind"], "metric_scaling")
            self.assertTrue(
                any(
                    row.get("run_id") == "run_live" and row.get("metric_value") == 0.123
                    for row in metric_plot["rows"]
                )
            )

    def test_pipeline_ui_declares_g2m_deeph_endpoints(self):
        source = (SCRIPTS_DIR / "pipeline_ui.py").read_text(encoding="utf-8")
        for endpoint in (
            "/api/g2m-deeph/validate-dataset",
            "/api/g2m-deeph/run",
            "/api/g2m-deeph/stop",
            "/api/g2m-deeph/status",
                "/api/g2m-deeph/logs",
                "/api/g2m-deeph/results",
                "/api/g2m-deeph/plot-runs",
                "/api/g2m-deeph/plots",
            ):
            self.assertIn(endpoint, source)

    def test_validate_dataset_http_endpoint_returns_summary(self):
        spec = importlib.util.spec_from_file_location(
            "pipeline_ui_g2m_deeph_endpoint_test",
            SCRIPTS_DIR / "pipeline_ui.py",
        )
        assert spec and spec.loader
        pipeline_ui = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pipeline_ui
        spec.loader.exec_module(pipeline_ui)

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            _write_dataset_provenance(dataset)
            _write_snapshot(dataset, "0", complete=True)
            server = ThreadingHTTPServer(("127.0.0.1", 0), pipeline_ui.ComparisonUIHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/g2m-deeph/validate-dataset"
                request = urllib.request.Request(
                    url,
                    data=json.dumps({"dataset_root": str(dataset)}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertTrue(payload["benchmark_ready"])
            self.assertEqual(payload["artifact_summary"]["total_snapshots"], 1)

    def _benchmark_validation_payload(self, dataset_root: Path) -> dict[str, object]:
        return {
            "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
            "benchmark_ready": True,
            "repair_required": False,
            "dataset_root": str(dataset_root),
            "snapshot_root": str(dataset_root),
            "artifact_summary": {},
            "errors": [],
            "warnings": [],
            "manifest_paths": {},
        }

    def _h_then_derivative_prediction_payload(
        self,
        result_dir: Path,
        *,
        derivative: dict | None = None,
        stages: dict | None = None,
    ) -> dict:
        stage_overrides = {
            "build_derivative_stencils": False,
            "validate_derivative_stencils": False,
            "run_derivative_siesta_reference": False,
            "derivative_metrics_graph2mat": False,
            "derivative_metrics_deeph": False,
            "derivative_gate_check": False,
            "derivative_plots": False,
        }
        stage_overrides.update(stages or {})
        derivative_config = {
            "enabled": True,
            "result_dir": str(result_dir),
            "method": "central",
            "deeph_command": "deeph-inference {stencil_root} {output_root} {model_dir}",
        }
        derivative_config.update(derivative or {})
        payload = {
            "workflow_mode": "h_then_derivative_full",
            "derivative": derivative_config,
            "stages": stage_overrides,
        }
        payload["modular_workflow"] = _normalized_modular_workflow_payload(payload)
        return payload

    def _benchmark_contexts(self, root: Path) -> tuple[Graph2MatBenchmarkContext, DeepHBenchmarkContext]:
        dataset_root = root / "dataset"
        run_root = root / "run"
        graph2mat_eval_root = run_root / "common_metrics" / "graph2mat_eval"
        deeph_eval_root = run_root / "common_metrics" / "deeph_eval"
        dataset_root.mkdir(parents=True, exist_ok=True)
        graph2mat_eval_root.mkdir(parents=True, exist_ok=True)
        deeph_eval_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps({"rows": [], "valid": True, "split_hash": "split-hash"}) + "\n",
            encoding="utf-8",
        )
        (dataset_root / "benchmark_dataset_manifest.json").write_text(
            json.dumps({"benchmark_ready": True, "generation_mode": "reused_validated", "warnings": []}) + "\n",
            encoding="utf-8",
        )
        training_dir = run_root / "graph2mat_training"
        training_dir.mkdir(parents=True, exist_ok=True)
        (training_dir / "checkpoint_manifest.json").write_text("{}\n", encoding="utf-8")
        graph_context = Graph2MatBenchmarkContext(
            dataset_root=dataset_root,
            run_root=run_root,
            graph2mat_root=root / "graph2mat",
            training_dir=training_dir,
            prediction_structs_dir=run_root / "graph2mat_predictions",
            config_path=run_root / "graph2mat_config.yaml",
            graph2mat_config_path=run_root / "graph2mat_config_resolved.yaml",
            graph2mat_manifest_path=run_root / "graph2mat" / "graph2mat_manifest.json",
            frozen_split_manifest_path=dataset_root / "frozen_split_manifest.json",
            benchmark_dataset_manifest_path=dataset_root / "benchmark_dataset_manifest.json",
            runs_json_path=training_dir / "runs.json",
            runs_json_counts={},
            train_glob="train/*.pkl",
            validation_glob="validation/*.pkl",
            predict_glob="test/*.pkl",
            output_file="predictions.npz",
            test_sample_ids=["s0"],
            split_hash="split-hash",
            prediction_split="test",
            dry_run=False,
        )
        deeph_context = DeepHBenchmarkContext(
            root=root / "deeph",
            raw_dir=root / "deeph_raw",
            processed_dir=root / "deeph_processed",
            graph_dir=root / "deeph_graph",
            save_dir=root / "deeph_save",
            inference_dir=root / "deeph_inference",
            preprocess_config=root / "deeph" / "preprocess.ini",
            train_config=root / "deeph" / "train.ini",
            inference_configs=[root / "deeph" / "inference_test.ini"],
            inference_work_dirs=[root / "deeph" / "inference_work"],
            manifest_path=run_root / "deeph" / "deeph_manifest.json",
            deeph_discovery={"source": "test"},
            split_audit_path=run_root / "deeph" / "split_audit.json",
            split_audit_csv_path=run_root / "deeph" / "split_audit.csv",
            split_hash="split-hash",
            raw_mirror={"rows": [{"sample_id": "s0", "split": "test", "raw_dir": str(root / "deeph_raw" / "s0")}]},
            inference_split="test",
            dry_run=False,
        )
        return graph_context, deeph_context

    def _run_common_metrics_workflow(
        self,
        *,
        derivative_enabled: bool,
        derivative_stencils_exist: bool,
        failing_derivative_method: str | None = None,
    ) -> tuple[Graph2MatDeepHBenchmarkRunner, list[list[str]], list[dict[str, object]]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_context, deeph_context = self._benchmark_contexts(root)
            runner = Graph2MatDeepHBenchmarkRunner()
            runner._state.running = True
            runner._state.started_at = time.time()
            recorded_commands: list[list[str]] = []
            aggregate_calls: list[dict[str, object]] = []

            def fake_run_command(command, *, cwd, env, label, allowed_returncodes=(0,), **kwargs):
                recorded_commands.append(list(command))
                if (
                    failing_derivative_method is not None
                    and "evaluate_hamiltonian_derivative_metrics.py" in " ".join(command)
                    and command[command.index("--source-model") + 1] == failing_derivative_method
                ):
                    raise CommandRunError(
                        f"{label} failed",
                        {
                            "label": label,
                            "command": list(command),
                            "cwd": str(cwd),
                            "started_at": 1.0,
                            "finished_at": 2.0,
                            "elapsed_seconds": 1.0,
                            "returncode": 2,
                        },
                    )
                return {
                    "label": label,
                    "command": list(command),
                    "cwd": str(cwd),
                    "started_at": 1.0,
                    "finished_at": 2.0,
                    "elapsed_seconds": 1.0,
                    "returncode": 0,
                }

            def fake_aggregate_common_metrics(**kwargs):
                aggregate_calls.append(dict(kwargs))
                return {
                    "status": "valid_reused_joint_dataset",
                    "warnings": [],
                    "summary_rows": [
                        {"method": "graph2mat", "h_mae_eV_mean": 0.2},
                        {"method": "deeph", "h_mae_eV_mean": 0.1},
                    ],
                    "derivative_summary_rows": [],
                    "derivative_metrics": {
                        "available": False,
                        "winner_metric": False,
                        "paper_level": False,
                        "summary_rows": [],
                    },
                    "recommendation": {
                        "winner": "deeph",
                        "robust_recommendation": True,
                        "primary_metric": "h_mae_eV_mean",
                    },
                }

            payload = {"run_id": "derivative-runner-test"}
            if derivative_enabled:
                payload["derivative_metrics"] = {"enabled": True}
            validation = self._benchmark_validation_payload(graph_context.dataset_root)

            with ExitStack() as stack:
                runner._run_command = fake_run_command  # type: ignore[method-assign]
                runner._run_ranking = mock.Mock(  # type: ignore[method-assign]
                    return_value={
                        "recommendation": {
                            "status": "exploratory_deeph_win",
                            "scientific_status": "exploratory_only",
                            "winner": "deeph",
                            "primary_metric": "h_mae_eV",
                        },
                        "best_runs_by_model": [],
                        "pairwise_graph2mat_vs_deeph": [],
                        "pareto_accuracy_cost": [],
                    }
                )
                stack.enter_context(mock.patch.object(runner, "_prepare_graph2mat_context", return_value=graph_context))
                stack.enter_context(mock.patch.object(runner, "_prepare_deeph_context", return_value=deeph_context))
                stack.enter_context(mock.patch.object(runner, "_graph2mat_python", return_value=sys.executable))
                stack.enter_context(mock.patch.object(runner, "_deeph_command", side_effect=lambda _payload, name: name))
                stack.enter_context(mock.patch.object(runner, "_audit_deeph_split", return_value={}))
                stack.enter_context(mock.patch.object(runner, "_stage_deeph_inference_inputs", return_value={"staged": True}))
                stack.enter_context(mock.patch.object(runner, "_validate_graph2mat_prediction_outputs", return_value={}))
                stack.enter_context(mock.patch.object(runner, "_validate_deeph_prediction_outputs", return_value={}))
                stack.enter_context(mock.patch.object(runner, "_validate_deeph_training_outputs", return_value={}))
                stack.enter_context(mock.patch.object(runner, "_write_graph2mat_manifest"))
                stack.enter_context(mock.patch.object(runner, "_write_deeph_manifest"))
                stack.enter_context(mock.patch.object(runner, "_write_run_cost_telemetry", return_value={}))
                stack.enter_context(
                    mock.patch(
                        "g2m_deeph_runner.stage_graph2mat_metric_result",
                        return_value=mock.Mock(result_dir=graph_context.run_root / "common_metrics" / "graph2mat_eval", sample_ids=["s0"]),
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "g2m_deeph_runner.stage_deeph_metric_inputs",
                        return_value=mock.Mock(
                            processed_dir=graph_context.run_root / "common_metrics" / "deeph_inputs" / "processed",
                            predictions_dir=graph_context.run_root / "common_metrics" / "deeph_inputs" / "predictions",
                            sample_ids=["s0"],
                        ),
                    )
                )
                stack.enter_context(mock.patch("g2m_deeph_runner.aggregate_common_metrics", side_effect=fake_aggregate_common_metrics))
                stack.enter_context(mock.patch("g2m_deeph_runner._force_diagnostic_metric_manifest"))
                stack.enter_context(mock.patch("g2m_deeph_runner._has_derivative_stencils", return_value=derivative_stencils_exist))
                stack.enter_context(
                    mock.patch(
                        "g2m_deeph_runner.write_derivative_plot_outputs",
                        return_value={
                            "payload": {"available": True, "plots": []},
                            "manifest": {"available": True},
                            "payload_path": graph_context.run_root / "common_metrics" / "summary" / "derivative_plots" / "derivative_plot_payload.json",
                            "manifest_path": graph_context.run_root / "common_metrics" / "summary" / "derivative_plots" / "derivative_plot_manifest.json",
                        },
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "g2m_deeph_runner.build_derivative_gate_report",
                        return_value={
                            "schema_version": "graph2mat_deeph_derivative_gate_report_v1",
                            "scientific_status": "internal_diagnostic",
                            "blockers": [],
                            "warnings": [],
                        },
                    )
                )
                runner._run_workflow(payload, validation, allow_repair=False)

            self.assertEqual(runner.status()["returncode"], 0)
            return runner, recorded_commands, aggregate_calls

    def test_runner_default_disabled_preserves_existing_common_metric_behavior(self):
        runner, recorded_commands, aggregate_calls = self._run_common_metrics_workflow(
            derivative_enabled=False,
            derivative_stencils_exist=True,
        )

        derivative_commands = [
            command for command in recorded_commands if any("evaluate_hamiltonian_derivative_metrics.py" in token for token in command)
        ]
        self.assertEqual(derivative_commands, [])
        self.assertEqual(len(aggregate_calls), 1)
        self.assertIsNone(aggregate_calls[0]["graph2mat_derivative_root"])
        self.assertIsNone(aggregate_calls[0]["deeph_derivative_root"])
        assert runner._last_results is not None
        derivative_metrics = runner._last_results["common_metrics"]["derivative_metrics"]
        self.assertFalse(derivative_metrics["enabled"])
        self.assertEqual(derivative_metrics["execution"], {})

    def test_runner_enabled_derivative_plumbing_records_commands_and_passes_roots(self):
        runner, recorded_commands, aggregate_calls = self._run_common_metrics_workflow(
            derivative_enabled=True,
            derivative_stencils_exist=True,
        )

        derivative_commands = [
            command for command in recorded_commands if any("evaluate_hamiltonian_derivative_metrics.py" in token for token in command)
        ]
        self.assertEqual(len(derivative_commands), 2)
        self.assertEqual(
            {command[command.index("--source-model") + 1] for command in derivative_commands},
            {"graph2mat", "deeph"},
        )
        self.assertEqual(len(aggregate_calls), 1)
        assert runner._last_results is not None
        common_root = Path(runner._last_results["graph2mat"]["run_root"]) / "common_metrics"
        self.assertEqual(aggregate_calls[0]["graph2mat_derivative_root"], common_root / "graph2mat_eval" / "derivative_metrics")
        self.assertEqual(aggregate_calls[0]["deeph_derivative_root"], common_root / "deeph_eval" / "derivative_metrics")
        derivative_metrics = runner._last_results["common_metrics"]["derivative_metrics"]
        self.assertTrue(derivative_metrics["enabled"])
        self.assertEqual(derivative_metrics["plot_outputs"]["status"], "completed")
        self.assertEqual(derivative_metrics["gate_report"]["status"], "completed")
        self.assertFalse(derivative_metrics["winner_metric"])
        self.assertEqual(runner._last_results["common_metrics"]["recommendation"]["primary_metric"], "h_mae_eV_mean")

    def test_runner_enabled_derivative_metrics_skip_when_no_stencils_exist(self):
        runner, recorded_commands, aggregate_calls = self._run_common_metrics_workflow(
            derivative_enabled=True,
            derivative_stencils_exist=False,
        )

        derivative_commands = [
            command for command in recorded_commands if any("evaluate_hamiltonian_derivative_metrics.py" in token for token in command)
        ]
        self.assertEqual(derivative_commands, [])
        self.assertEqual(len(aggregate_calls), 1)
        assert runner._last_results is not None
        derivative_execution = runner._last_results["common_metrics"]["derivative_metrics"]["execution"]
        self.assertEqual(
            {record["status"] for record in derivative_execution.values()},
            {"skipped_no_stencils"},
        )

    def test_runner_derivative_failures_are_reported_without_overwriting_h_metrics(self):
        runner, recorded_commands, _aggregate_calls = self._run_common_metrics_workflow(
            derivative_enabled=True,
            derivative_stencils_exist=True,
            failing_derivative_method="graph2mat",
        )

        derivative_commands = [
            command for command in recorded_commands if any("evaluate_hamiltonian_derivative_metrics.py" in token for token in command)
        ]
        self.assertEqual(len(derivative_commands), 2)
        assert runner._last_results is not None
        common_metrics = runner._last_results["common_metrics"]
        self.assertEqual(common_metrics["recommendation"]["winner"], "deeph")
        self.assertEqual(common_metrics["derivative_metrics"]["execution"]["graph2mat"]["returncode"], 2)
        self.assertEqual(common_metrics["runs"]["derivative_metrics"]["graph2mat"]["returncode"], 2)
        self.assertIn("existing H metrics are preserved", "".join(runner.logs()["lines"]))

    def test_backfill_derivative_postprocess_uses_only_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_run_root = root / "benchmark"
            sweep_root = benchmark_run_root / "sweep"
            summary_root = benchmark_run_root / "summary"
            summary_root.mkdir(parents=True, exist_ok=True)
            completed_graph = benchmark_run_root / "sweep" / "graph2mat" / "done_graph"
            completed_deeph = benchmark_run_root / "sweep" / "deeph" / "done_deeph"
            failed_root = benchmark_run_root / "sweep" / "deeph" / "failed_deeph"
            (completed_graph / "metrics" / "graph2mat" / "eval_input").mkdir(parents=True, exist_ok=True)
            (completed_deeph / "metrics" / "deeph" / "eval").mkdir(parents=True, exist_ok=True)
            training_sweep_manifest = sweep_root / "training_sweep_manifest.json"
            training_sweep_manifest.parent.mkdir(parents=True, exist_ok=True)
            training_sweep_manifest.write_text(
                json.dumps(
                    {
                        "runs": [
                            {"status": "completed", "run_root": str(completed_graph)},
                            {"status": "completed", "run_root": str(completed_deeph)},
                            {"status": "failed", "run_root": str(failed_root)},
                            {"status": "completed"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            processed: list[Path] = []

            def fake_run_derivative_postprocess(**kwargs):
                processed.append(Path(kwargs["run_root"]))
                return {
                    "enabled": True,
                    "settings": kwargs["settings"],
                    "execution": {
                        "graph2mat": {"status": "completed" if "graph2mat" in str(kwargs["run_root"]) else "skipped_missing_input"},
                        "deeph": {"status": "completed" if "deeph" in str(kwargs["run_root"]) else "skipped_missing_input"},
                    },
                    "plot_outputs": {"status": "completed"},
                    "gate_report": {"status": "completed"},
                    "roots": {"graph2mat": "", "deeph": ""},
                }

            with mock.patch("g2m_deeph_runner.run_derivative_postprocess", side_effect=fake_run_derivative_postprocess):
                summary = backfill_derivative_postprocess_from_training_sweep(
                    training_sweep_manifest_path=training_sweep_manifest,
                    settings={
                        "enabled": True,
                        "finite_difference_method": "central",
                        "split": "test",
                        "require_central": True,
                        "diagnostic_only": True,
                        "support_threshold": 1e-12,
                        "overwrite": True,
                    },
                    python_executable=sys.executable,
                )

            self.assertEqual(processed, [completed_graph.resolve(), completed_deeph.resolve()])
            self.assertEqual(summary["processed_runs"], 2)
            self.assertTrue((summary_root / "derivative_backfill_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
