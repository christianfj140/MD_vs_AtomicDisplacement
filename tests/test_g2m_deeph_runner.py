import json
import importlib.util
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
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
    METRIC_FAIL_POLICY_DIAGNOSTIC_ONLY,
    METRIC_FAIL_POLICY_FAIL_CLOSED,
    Graph2MatDeepHBenchmarkRunner,
    _deeph_training_parallelism,
    _deeph_metric_command_args,
    _extract_validation_metrics,
    _force_diagnostic_metric_manifest,
    _link_or_copy_file,
    _metric_allowed_returncodes,
    _metric_evaluation_split,
    _metric_fail_policy,
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
            self.assertEqual(payload["metric_scaling_rows"], [live_row])
            self.assertEqual(payload["live_metric_rows"], 1)
            self.assertEqual(payload["archived_runs"], 0)
            metric_plot = next(plot for plot in payload["plots"] if plot["id"] == "metric_scaling_h_mae")
            self.assertEqual(metric_plot["kind"], "metric_scaling")
            self.assertEqual(metric_plot["rows"][0]["metric_value"], 0.123)

    def test_pipeline_ui_declares_g2m_deeph_endpoints(self):
        source = (SCRIPTS_DIR / "pipeline_ui.py").read_text(encoding="utf-8")
        for endpoint in (
            "/api/g2m-deeph/validate-dataset",
            "/api/g2m-deeph/run",
            "/api/g2m-deeph/stop",
            "/api/g2m-deeph/status",
            "/api/g2m-deeph/logs",
            "/api/g2m-deeph/results",
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


if __name__ == "__main__":
    unittest.main()
