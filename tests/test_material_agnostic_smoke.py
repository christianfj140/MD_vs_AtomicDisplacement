from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Comparison" / "scripts" / "material_agnostic_smoke.py"


def load_smoke_module():
    for path in (
        REPO_ROOT / "Comparison" / "scripts",
        REPO_ROOT / "shared",
        REPO_ROOT / "AtomDisplacement" / "scripts",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "material_agnostic_smoke_test",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MaterialAgnosticSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.module = load_smoke_module()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_h2o_preset_dry_run_smoke(self) -> None:
        report = self.module.run_smoke(
            cases=["h2o"],
            output_root=self.root / "smoke",
            include_failure_checks=False,
        )

        self.assertTrue(report["ok"])
        self.assertFalse(report["external_execution_required"])
        case = report["cases"][0]
        self.assertEqual(case["material"]["label"], "h2o")
        self.assertEqual(case["material"]["atom_count"], 3)
        self.assertEqual(case["generic_cartesian"]["generated_structures"], 18)
        self.assertEqual(case["generic_random_cartesian"]["generated_structures"], 3)
        self.assertTrue(case["generic_random_cartesian"]["deterministic"])
        self.assertEqual(case["graph2mat_config"]["matrix_target"], "hamiltonian")
        self.assertEqual(case["material_provenance"]["material_label"], "h2o")
        self.assertTrue((self.root / "smoke" / "h2o" / "smoke_manifest.json").exists())

    def test_synthetic_non_h2o_material_dry_run_smoke(self) -> None:
        report = self.module.run_smoke(
            cases=["synthetic"],
            output_root=self.root / "smoke",
            include_failure_checks=False,
        )

        case = report["cases"][0]
        self.assertEqual(case["material"]["label"], "sic")
        self.assertEqual(case["material"]["atom_count"], 2)
        self.assertEqual([item["label"] for item in case["material"]["species"]], ["Si", "C"])
        self.assertEqual(case["generic_cartesian"]["generated_structures"], 12)
        self.assertEqual(case["generic_random_cartesian"]["split_strategy"], "grouped_family_round_robin")
        self.assertIn("material_basis/*.ion.xml", case["graph2mat_config"]["basis_files"])
        self.assertEqual(case["material_provenance"]["material_label"], "sic")
        self.assertIn("fdf_sha256", case["material_provenance"])

    def test_missing_pseudo_and_unsupported_fdf_fail_clearly(self) -> None:
        failures = self.module.expected_failure_checks(self.root / "failures")
        by_case = {item["case"]: item for item in failures}

        self.assertIn("Missing pseudopotential for species 'C'", by_case["missing_pseudo"]["message"])
        self.assertIn("unsupported AtomicCoordinatesFormat", by_case["unsupported_fdf"]["message"])

    def test_smoke_unit_path_requires_no_real_siesta_or_graph2mat(self) -> None:
        report = self.module.run_smoke(
            cases=["synthetic"],
            output_root=self.root / "smoke",
            include_failure_checks=True,
        )

        self.assertFalse(report["external_execution_required"])
        self.assertTrue(report["expected_failure_checks"])
        script_text = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("subprocess.run", script_text)
        self.assertNotIn("graph2mat models", script_text)

    def test_cli_smoke_runs_without_external_executables(self) -> None:
        output_dir = self.root / "cli_smoke"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--case",
                "synthetic",
                "--output-dir",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["external_execution_required"])
        self.assertTrue((output_dir / "material_agnostic_smoke_report.json").exists())


if __name__ == "__main__":
    unittest.main()
