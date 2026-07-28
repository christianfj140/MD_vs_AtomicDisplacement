from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Comparison" / "scripts" / "g2m_deeph_smoke.py"
DOC_PATH = REPO_ROOT / "docs" / "graph2mat_deeph_benchmark.md"


def load_smoke_module():
    scripts_dir = REPO_ROOT / "Comparison" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "g2m_deeph_smoke_test",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Graph2MatDeepHSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.module = load_smoke_module()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dry_run_smoke_creates_planned_execution_manifest(self) -> None:
        output_root = self.root / "smoke"
        manifest = self.module.run_smoke(
            output_root=output_root,
            dry_run=True,
            tiny_real=False,
            sample_limit=6,
            run_id="unit_smoke",
            timeout_seconds=20,
        )

        self.assertEqual(manifest["status"], "dry_run_passed")
        self.assertTrue(manifest["ok"])
        self.assertFalse(manifest["skip"])
        for name in (
            "smoke_manifest.json",
            "artifact_validation.json",
            "benchmark_manifest.json",
            "recommendation.json",
            "logs/smoke.log",
        ):
            self.assertTrue((output_root / name).exists(), name)

        saved = json.loads((output_root / "smoke_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "dry_run_passed")
        self.assertEqual(saved["dataset_generation_returncode"], 0)
        self.assertIn("SystemLabel.HSX", saved["required_snapshot_artifacts"])

    def test_missing_dependencies_produce_skip_not_pass(self) -> None:
        output_root = self.root / "real_skip"
        with mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
            manifest = self.module.run_smoke(
                output_root=output_root,
                dry_run=False,
                tiny_real=True,
                sample_limit=6,
                run_id="unit_real_skip",
            )

        self.assertEqual(manifest["status"], "skipped")
        self.assertIsNone(manifest["ok"])
        self.assertTrue(manifest["skip"])
        self.assertTrue(manifest["skip_reasons"])
        self.assertIn("RUN_G2M_DEEPH_REAL_SMOKE=1", " ".join(manifest["skip_reasons"]))

    def test_tiny_real_smoke_is_env_gated(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
            report = self.module._dependency_report({})

        self.assertEqual(report["status"], "missing")
        self.assertFalse(report["checks"]["RUN_G2M_DEEPH_REAL_SMOKE"]["available"])

    def test_smoke_validation_lists_required_artifact_names(self) -> None:
        artifacts = self.module.required_snapshot_artifacts()

        for artifact in (
            "RUN.fdf",
            "SystemLabel.TSHS",
            "SystemLabel.TSDE",
            "SystemLabel.HSX",
            "SystemLabel.STRUCT_OUT",
            "SystemLabel.XV",
            "SystemLabel.ORB_INDX",
            "metadata.json",
        ):
            self.assertIn(artifact, artifacts)

    def test_smoke_cli_dry_run(self) -> None:
        output_root = self.root / "cli_smoke"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--dry-run",
                "--output-root",
                str(output_root),
                "--run-id",
                "unit_cli_smoke",
                "--timeout-seconds",
                "20",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "dry_run_passed")
        self.assertTrue((output_root / "smoke_manifest.json").exists())

    def test_paper_workflow_dry_run_writes_blocked_control_plane_summary(self) -> None:
        output_root = self.root / "paper_smoke"
        summary = self.module.run_paper_workflow_dry_run(
            output_root=output_root,
            run_id="unit_paper_smoke",
        )

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["status"], "blocked_as_expected")
        self.assertEqual(summary["scientific_status"], "not_a_scientific_run")
        self.assertFalse(summary["robust_claim_allowed"])
        self.assertTrue(summary["diagnostic_only"])
        self.assertTrue((output_root / "smoke_summary.json").exists())
        self.assertFalse(Path(summary["outputs"]["search_plan"]).exists())
        self.assertTrue(Path(summary["outputs"]["gate_status"]).exists())
        self.assertTrue(Path(summary["outputs"]["release_manifest"]).exists())

        gate_status = json.loads(Path(summary["outputs"]["gate_status"]).read_text(encoding="utf-8"))
        self.assertFalse(gate_status["robust_claim_allowed"])
        self.assertIn(
            gate_status["claim_status"],
            {"diagnostic_only", "invalid_equivalence", "invalid_missing_evidence"},
        )

    def test_smoke_cli_paper_workflow_dry_run(self) -> None:
        output_root = self.root / "cli_paper_smoke"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--paper-workflow-dry-run",
                "--output-dir",
                str(output_root),
                "--run-id",
                "unit_cli_paper_smoke",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scientific_status"], "not_a_scientific_run")
        self.assertFalse(payload["robust_claim_allowed"])
        self.assertTrue((output_root / "smoke_summary.json").exists())
        self.assertTrue((output_root / "paper_workflow" / "gate_status.json").exists())

    def test_smoke_cli_help_mentions_paper_workflow_dry_run(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
        self.assertIn("--paper-workflow-dry-run", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_smoke_command_is_documented(self) -> None:
        doc = DOC_PATH.read_text(encoding="utf-8")

        self.assertIn("g2m_deeph_smoke.py", doc)
        self.assertIn("RUN_G2M_DEEPH_REAL_SMOKE", doc)
        self.assertIn("--dry-run", doc)
        self.assertIn("--tiny-real", doc)
        self.assertIn("--paper-workflow-dry-run", doc)
        self.assertIn("not_a_scientific_run", doc)


if __name__ == "__main__":
    unittest.main()
